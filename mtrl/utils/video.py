# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Implementation based on Denis Yarats' implementation of [SAC](https://github.com/denisyarats/pytorch_sac).
"""Utility to record the environment frames into a video."""
import os

import imageio


class VideoRecorder(object):
    def __init__(self, dir_name, height=256, width=256, camera_id=0, fps=30):
        """Class to record the environment frames into a video.

        Args:
            dir_name ([type]): directory to save the recording.
            height (int, optional): height of the frame. Defaults to 256.
            width (int, optional): width of the frame. Defaults to 256.
            camera_id (int, optional): id of the camera for recording. Defaults to 0.
            fps (int, optional): frames-per-second for the recording. Defaults to 30.
        """
        self.dir_name = dir_name
        self.height = height
        self.width = width
        self.camera_id = camera_id
        self.fps = fps
        self.frames = []

    def init(self, enabled=True):
        """Initialize the recorder.

        Args:
            enabled (bool, optional): should enable the recorder or not. Defaults to True.
        """
        self.frames = []
        self.enabled = self.dir_name is not None and enabled

    def record(self, frame, env=None):
        """Record the frames.

        Args:
            env ([type]): environment to record the frames.
        """
        if self.enabled:
            if frame is None:
                assert env is not None
                frame = env.render(
                    mode="rgb_array",
                    height=self.height,
                    width=self.width,
                    camera_id=self.camera_id,
                )
            self.frames.append(frame)

    def save(self, file_name):
        """Save the frames as video to `self.dir_name` in a file named `file_name`.

        Args:
            file_name ([type]): name of the file to store the video frames.
        """
        if self.enabled:
            path = os.path.join(self.dir_name, file_name)
            imageio.mimsave(path, self.frames, fps=self.fps)
