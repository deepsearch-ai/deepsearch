import unittest

from deepsearchai.embedding_models.whisper_model import Whisper


class TestGetWhisperMediaEncoding(unittest.TestCase):
    def test_audio_encoding(self):
        """Test that the `get_media_encoding()` function can correctly identify the encoding of an audio file."""

        whisper = Whisper()
        # Load the audio file.
        audio_data = open("audio.ogg", "rb").read()

        # Get the media encoding.
        media_encoding = whisper.get_media_encoding(
            audio_data, whisper.MEDIA_TYPE.AUDIO
        )

        # Assert that the media encoding is correct.
        self.assertEqual(media_encoding, "audio/ogg;codecs=opus")

    def test_video_encoding(self):
        """Test that the `get_media_encoding()` function can correctly identify the encoding of a video file."""

        whisper = Whisper()
        # Load the video file.
        video_data = open("video.mp4", "rb").read()

        # Get the media encoding.
        media_encoding = whisper.get_media_encoding(
            video_data, whisper.MEDIA_TYPE.VIDEO
        )

        # Assert that the media encoding is correct.
        self.assertEqual(media_encoding, "video/mp4;codecs=h264")

    def test_invalid_media_type(self):
        """Test that the `get_media_encoding()` function raises a ValueError if an invalid media type is specified."""

        whisper = Whisper()
        # Get the media encoding.
        with self.assertRaises(ValueError):
            whisper.get_media_encoding(b"", "invalid_media_type")
