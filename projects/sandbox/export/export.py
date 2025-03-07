import logging
from pathlib import Path
from typing import Callable, Optional

import torch

import hermes.quiver as qv
from bbhnet.architectures import Preprocessor, architecturize
from bbhnet.logging import configure_logging


def scale_model(model, instances):
    # TODO: should quiver handle this under the hood?
    try:
        model.config.scale_instance_group(instances)
    except ValueError:
        model.config.add_instance_group(count=instances)


@architecturize
def export(
    architecture: Callable,
    repository_directory: str,
    outdir: Path,
    num_ifos: int,
    inference_sampling_rate: float,
    sample_rate: float,
    batch_size: int,
    fduration: Optional[float] = None,
    weights: Optional[Path] = None,
    streams_per_gpu: int = 1,
    bbhnet_instances: Optional[int] = None,
    preproc_instances: Optional[int] = None,
    platform: qv.Platform = qv.Platform.ONNX,
    clean: bool = False,
    verbose: bool = False,
) -> None:
    """
    Export a BBHNet architecture to a model repository
    for streaming inference, including adding a model
    for caching input snapshot state on the server.

    Args:
        architecture:
            A function which takes as input a number of witness
            channels and returns an instantiated torch `Module`
            which represents a DeepClean network architecture
        repository_directory:
            Directory to which to save the models and their
            configs
        outdir:
            Path to save logs. If `weights` is `None`, this
            directory is assumed to contain a file `"weights.pt"`.
        num_ifos:
            The number of interferometers contained along the
            channel dimension used to train BBHNet
        inference_sampling_rate:
            The rate at which kernels are sampled from the
            h(t) timeseries. This, along with the `sample_rate`,
            dictates the size of the update expected at the
            snapshotter model
        sample_rate:
            Rate at which the input kernel has been sampled, in Hz
        weights:
            Path to a set of trained weights with which to
            initialize the network architecture. If left as
            `None`, a file called `"weights.pt"` will be looked
            for in the `output_directory`.
        streams_per_gpu:
            The number of snapshot states to host per GPU during
            inference
        instances:
            The number of concurrent execution instances of the
            BBHNet architecture to host per GPU during inference
        platform:
            The backend framework platform used to host the
            DeepClean architecture on the inference service. Right
            now only `"onnxruntime_onnx"` is supported.
        clean:
            Whether to clear the repository directory before starting
            export
        verbose:
            If set, log at `DEBUG` verbosity, otherwise log at
            `INFO` verbosity.
        **kwargs:
            key word arguments specific to the export platform
    """

    # make relevant directories
    outdir.mkdir(exist_ok=True, parents=True)

    # if we didn't specify a weights filename, assume
    # that a "weights.pt" lives in our output directory
    if weights is None or weights.is_dir():
        weights_dir = outdir if weights is None else weights
        weights = weights_dir / "weights.pt"
    if not weights.exists():
        raise FileNotFoundError(f"No weights file '{weights}'")

    configure_logging(outdir / "export.log", verbose)

    # instantiate a new, randomly initialized version
    # of the network architecture, including preprocessor
    logging.info("Initializing model architecture")
    nn = architecture(num_ifos)
    preprocessor = Preprocessor(num_ifos, sample_rate, fduration=fduration)
    nn = torch.nn.Sequential(preprocessor, nn)
    logging.info(f"Initialize:\n{nn}")

    # load in a set of trained weights and use the
    # preprocessor's kernel_length parameter to infer
    # the expected input dimension along the time axis
    logging.info(f"Loading parameters from {weights}")
    state_dict = torch.load(weights, map_location="cpu")
    nn.load_state_dict(state_dict)
    kernel_length = preprocessor.whitener.kernel_length.item()
    logging.info(
        f"Model will be exported with input kernel length of {kernel_length}s"
    )
    nn.eval()

    # instantiate a model repository at the
    # indicated location. Split up the preprocessor
    # and the neural network (which we'll call bbhnet)
    # to export/scale them separately, and start by
    # seeing if either already exists in the model repo
    repo = qv.ModelRepository(repository_directory, clean)
    try:
        bbhnet = repo.models["bbhnet"]
    except KeyError:
        bbhnet = repo.add("bbhnet", platform=platform)

    # do the same for preprocessor
    try:
        preproc = repo.models["preproc"]
    except KeyError:
        preproc = repo.add("preproc", platform=platform)

    # if we specified a number of instances we want per-gpu
    # for each model at inference time, scale them now
    if bbhnet_instances is not None:
        scale_model(bbhnet, bbhnet_instances)
    if preproc_instances is not None:
        scale_model(preproc, preproc_instances)

    # start by exporting the preprocessor, then  use
    # its inferred output shape to export the network
    input_dim = int(kernel_length * sample_rate)
    input_shape = (batch_size, num_ifos, input_dim)
    preproc.export_version(
        nn._modules["0"],
        input_shapes={"hoft": input_shape},
        output_names=["whitened"],
    )

    # the network will have some different keyword
    # arguments required for export depending on
    # the target inference platform
    # TODO: hardcoding these kwargs for now, but worth
    # thinking about a more robust way to handle this
    kwargs = {}
    if platform == qv.Platform.ONNX:
        kwargs["opset_version"] = 13

        # turn off graph optimization because of this error
        # https://github.com/triton-inference-server/server/issues/3418
        bbhnet.config.optimization.graph.level = -1
    elif platform == qv.Platform.TENSORRT:
        kwargs["use_fp16"] = True

    input_shape = tuple(preproc.config.output[0].dims)
    bbhnet.export_version(
        nn._modules["1"],
        input_shapes={"whitened": input_shape},
        output_names=["discriminator"],
        **kwargs,
    )

    # now try to create an ensemble that has a snapshotter
    # at the front for streaming new data to
    ensemble_name = "bbhnet-stream"
    stream_size = int(sample_rate / inference_sampling_rate)

    # see if we have an existing snapshot model up front,
    # since we'll make different choices later depending
    # on whether it already exists or not
    snapshotter = repo.models.get("snapshotter", None)

    try:
        # first see if we have an existing
        # ensemble with the given name
        ensemble = repo.models[ensemble_name]
    except KeyError:
        # if we don't, create one
        ensemble = repo.add(ensemble_name, platform=qv.Platform.ENSEMBLE)

        # if a snapshot model already exists, add it to
        # the ensemble and pipe its output to the input of
        # bbhnet, otherwise use the `add_streaming_inputs`
        # method on the ensemble to create a snapshotter
        # and perform this piping for us
        if snapshotter is not None:
            ensemble.add_input(snapshotter.inputs["stream"])
            ensemble.pipe(
                snapshotter.outputs["snapshotter"], preproc.inputs["hoft"]
            )
        else:
            # there's no snapshotter, so make one
            ensemble.add_streaming_inputs(
                inputs=[preproc.inputs["hoft"]],
                stream_size=stream_size,
                batch_size=batch_size,
                name="snapshotter",
                streams_per_gpu=streams_per_gpu,
            )
            snapshotter = repo.models["snapshotter"]

        # now send the outputs of the preprocessing module
        # to the inputs of the neural network
        ensemble.pipe(preproc.outputs["whitened"], bbhnet.inputs["whitened"])

        # export the ensemble model, which basically amounts
        # to writing its config and creating an empty version entry
        ensemble.add_output(bbhnet.outputs["discriminator"])
        ensemble.export_version(None)
    else:
        # if there does already exist an ensemble by
        # the given name, make sure it has BBHNet
        # and the snapshotter as a part of its models
        if bbhnet not in ensemble.models:
            raise ValueError(
                "Ensemble model '{}' already in repository "
                "but doesn't include model 'bbhnet'".format(ensemble_name)
            )
        elif snapshotter is None or snapshotter not in ensemble.models:
            raise ValueError(
                "Ensemble model '{}' already in repository "
                "but doesn't include model 'snapshotter'".format(ensemble_name)
            )

    # keep snapshot states around for a long time in case there are
    # unexpected bottlenecks which throttle update for a few seconds
    snapshotter.config.sequence_batching.max_sequence_idle_microseconds = int(
        6e10
    )
    snapshotter.config.write()


if __name__ == "__main__":
    export()
