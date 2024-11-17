from dataclasses import dataclass
from typing import List

import numpy as np
from manim import (
    BLUE,
    DOWN,
    LEFT,
    RED,
    RIGHT,
    UP,
    YELLOW,
    Arrow,
    Create,
    FadeOut,
    Rectangle,
    Scene,
    Text,
    VGroup,
    Write,
)


@dataclass
class TimelineConfig:
    start: float = -5
    end: float = 5
    vertical_position: float = -3
    right_shift: float = 1.0


@dataclass
class GPUConfig:
    height: float = 0.4
    vertical_spacing: float = 0.5
    animation_speed: float = 0.25


@dataclass
class AlgorithmConfig:
    label_right_margin: float = 7
    first_algo_vertical_pos: float = 2.5
    vertical_spacing: float = 1.2
    target_latency: float = 2.0
    speculation_width: float = 1.0
    verification_width: float = 1.0
    num_units: int = 4
    num_dsi_gpus: int = 8


class Colors:
    blue = BLUE
    yellow = YELLOW


@dataclass
class TimeUnit:
    width: float
    color: None | Colors


@dataclass
class GPU:
    time_units: List[TimeUnit]


@dataclass
class Algorithm:
    name: str
    gpus: List[GPU]


class DistributedSpeculativeInference(Scene):
    def __init__(self):
        super().__init__()
        self.timeline_config = TimelineConfig()
        self.gpu_config = GPUConfig()
        self.algo_config = AlgorithmConfig()

    def create_timeline(self, right_shift: float = None) -> VGroup:
        if right_shift is None:
            right_shift = self.timeline_config.right_shift

        timeline = (
            Arrow(
                start=LEFT * abs(self.timeline_config.start),
                end=RIGHT * self.timeline_config.end,
                buff=0,
                stroke_width=2,
                tip_length=0.2,
            )
            .shift(UP * self.timeline_config.vertical_position)
            .shift(RIGHT * right_shift)
        )
        timeline_label = Text("Timeline").next_to(timeline, DOWN)
        return VGroup(timeline, timeline_label)

    def create_gpu_rectangle(
        self, time_unit: TimeUnit, position: np.array
    ) -> Rectangle:
        """Create a rectangle representing a GPU computation unit"""
        if time_unit.color is None:
            return None
        rect = Rectangle(
            width=time_unit.width,
            height=self.gpu_config.height,
            fill_color=time_unit.color,
            fill_opacity=0.8,
        )
        # Align the rectangle's left edge with the given x position
        rect.align_to(position, LEFT)
        # Set the y position
        rect.set_y(position[1])
        return rect

    def create_algorithm_label(self, name: str, vertical_position: float) -> Text:
        """Create and position an algorithm name label"""
        label = Text(f"{name}:").align_to(
            LEFT * self.algo_config.label_right_margin, LEFT
        )
        label.shift(UP * vertical_position)
        return label

    def animate_algorithm(
        self, algorithm: Algorithm, vertical_position: float, start_x: float
    ) -> None:
        """Animate a single algorithm's execution"""
        algo_label = self.create_algorithm_label(algorithm.name, vertical_position)
        self.play(Write(algo_label))

        # First, collect all rectangles grouped by their starting x position
        rectangles_by_x = {}
        for i, gpu in enumerate(algorithm.gpus):
            current_x = start_x
            current_y = vertical_position - (i * self.gpu_config.vertical_spacing)

            for time_unit in gpu.time_units:
                if time_unit.color is not None:
                    rect = self.create_gpu_rectangle(
                        time_unit, np.array([current_x, current_y, 0])
                    )
                    if current_x not in rectangles_by_x:
                        rectangles_by_x[current_x] = []
                    rectangles_by_x[current_x].append(rect)
                current_x += time_unit.width

        # Animate rectangles in order of their x positions
        for x in sorted(rectangles_by_x.keys()):
            self.play(
                *[Create(rect) for rect in rectangles_by_x[x]],
                run_time=self.gpu_config.animation_speed,
            )

    def construct(self):
        # Create timeline
        timeline_group = self.create_timeline()
        self.play(Create(timeline_group))

        # Get the actual position of the timeline's origin
        timeline_start = timeline_group[
            0
        ].get_start()  # Get the start point of the line

        # Add a red reference rectangle at the exact timeline origin
        reference_rect = Rectangle(
            width=0.2, height=5, fill_color=RED, fill_opacity=0.8
        ).align_to(timeline_start, LEFT)
        self.play(Create(reference_rect))

        # Define algorithms
        non_si = Algorithm(
            name="Non-SI",
            gpus=[
                GPU(
                    time_units=[
                        TimeUnit(
                            width=self.algo_config.target_latency, color=Colors.blue
                        )
                    ]
                    * self.algo_config.num_units
                )
            ],
        )

        si = Algorithm(
            name="SI",
            gpus=[
                GPU(
                    time_units=[
                        TimeUnit(
                            width=self.algo_config.speculation_width,
                            color=Colors.yellow,
                        )
                    ]
                    * self.algo_config.num_units
                ),
                GPU(
                    time_units=[
                        TimeUnit(
                            width=self.algo_config.verification_width, color=Colors.blue
                        )
                    ]
                    * self.algo_config.num_units
                ),
            ],
        )

        dsi = Algorithm(
            name="DSI",
            gpus=[
                GPU(
                    time_units=[
                        *[
                            TimeUnit(
                                width=self.algo_config.speculation_width,
                                color=Colors.yellow,
                            )
                            for _ in range(self.algo_config.num_units)
                        ],
                        *[
                            TimeUnit(
                                width=self.algo_config.verification_width,
                                color=Colors.blue,
                            )
                            for _ in range(self.algo_config.num_units)
                        ],
                    ]
                )
                for _ in range(self.algo_config.num_dsi_gpus)
            ],
        )

        # Calculate vertical positions and animate
        current_vertical_pos = self.algo_config.first_algo_vertical_pos
        self.animate_algorithm(non_si, current_vertical_pos, timeline_start[0])
        current_vertical_pos -= self.algo_config.vertical_spacing
        self.animate_algorithm(si, current_vertical_pos, timeline_start[0])
        current_vertical_pos -= self.algo_config.vertical_spacing
        self.animate_algorithm(dsi, current_vertical_pos, timeline_start[0])

        # Finish animation
        self.wait(2)
        self.play(FadeOut(*self.mobjects))
