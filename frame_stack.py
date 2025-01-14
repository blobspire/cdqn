from collections import deque
import numpy as np
import cv2

# A stack of frames to store sequential pixel data to infer speed and direction of objects
class FrameStack:
    def __init__(self, stack_size):
        self.stack_size = stack_size
        self.frames = deque(maxlen=stack_size)

    # Fill the stack with the initial frame
    def reset(self, initial_frame):
        self.frames = deque([initial_frame] * self.stack_size, maxlen=self.stack_size)
        return np.stack(self.frames, axis=0)

    # Add the new frame to the stack
    def append(self, frame):
        self.frames.append(frame)
        return np.stack(self.frames, axis=0)
    
# Preprocess frame (decrease size and convert to grayscale)
def preprocess_frame(frame):
    # Resize to 84x84 and convert to grayscale
    frame = cv2.resize(frame, (84, 84))
    # Normalize pixel values to [0, 1]
    frame = np.array(frame, dtype=np.float32) / 255.0
    return frame