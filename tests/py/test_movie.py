import numpy as np
import pytest
from vertexcomp import movie


def test_frame_sync_options_match_ffmpeg_version(monkeypatch):
    monkeypatch.setattr(movie, "FFMPEG_VERSION", "4.4.4")
    assert movie._frame_sync_args() == ["-vsync", "0"]

    monkeypatch.setattr(movie, "FFMPEG_VERSION", "9.0.2")
    assert movie._frame_sync_args() == ["-fps_mode", "passthrough"]


@pytest.mark.skipif(not movie.available(), reason="imageio-ffmpeg is unavailable")
def test_read_decodes_movie_frames(tmp_path):
    path = str(tmp_path / "read-test.mp4")
    writer = movie.MovieWriter(path, 64, 48, 24)
    for level in (0.0, 0.5, 1.0):
        writer.write(np.full((48, 64, 3), level, np.float32))
    writer.close()

    provider = movie.MovieProvider()
    assert provider.info(path) == (3, 64, 48, 24.0)
    frames = [provider.frame(path, frame) for frame in range(3)]
    assert all(frame is not None for frame in frames)

    means = [float(frame[..., 0].mean()) for frame in frames]
    assert means[0] < 0.1
    assert 0.35 < means[1] < 0.65
    assert means[2] > 0.85
