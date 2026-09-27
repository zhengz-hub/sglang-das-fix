from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import unittest

from sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase


class _LegacyServing(OpenAIServingBase):
    def __init__(self, message):
        self.message = message

    def _request_id_prefix(self):
        return "test-"

    def _convert_to_internal_request(self, request, raw_request=None):
        raise NotImplementedError

    def _validate_request(self, request):
        return self.message


class RequestValidationDetailTestCase(unittest.TestCase):
    def test_default_delegate_wraps_legacy_validation_message(self):
        detail = _LegacyServing("legacy validation error")._validate_request_detail(
            object()
        )

        self.assertEqual(detail.message, "legacy validation error")
        self.assertIsNone(detail.param)

    def test_default_delegate_preserves_success(self):
        self.assertIsNone(_LegacyServing(None)._validate_request_detail(object()))


if __name__ == "__main__":
    unittest.main()
