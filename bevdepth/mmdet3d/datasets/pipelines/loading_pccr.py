# Gen2E: PCCR-specific input handling on top of the restored upstream BEVDet classes.
from ..builder import PIPELINES
from .loading_upstream import PrepareImageInputs as PrepareImageInputsUpstream


@PIPELINES.register_module()
class PrepareImageInputsPCCR(PrepareImageInputsUpstream):
    """Upstream BEVDet dev2.0 PrepareImageInputs plus `data_config['cams'] = None`.

    PCCR rigs have 4 to 10 cameras with rig-specific names. The official PCCR BEVDet
    config sets `cams=None, Ncams=None`, but the pipeline code that handles it is not
    in the PCCR repository; upstream `choose_cams()` fails on `len(None)`. With
    `cams=None` this uses every camera of the sample, in the order of `info['cams']`.
    No camera is dropped in training. All other behaviour is upstream.

    Use `ego_cam=None` with PCCR: upstream defaults to 'CAM_FRONT', which most rigs
    lack. With None each camera uses its own ego pose. That pose is identical across a
    PCCR sample's cameras, because CARLA captures all sensors on the same tick.
    """

    def get_inputs(self, results, flip=None, scale=None):
        self._sample_cams = list(results['curr']['cams'].keys())
        return super().get_inputs(results, flip=flip, scale=scale)

    def choose_cams(self):
        if self.data_config['cams'] is None:
            return self._sample_cams
        return super().choose_cams()
