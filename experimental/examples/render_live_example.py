import random
import time

from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.utils.rendertools import RenderTool


def main() -> None:
    env = RailEnv(
        width=30,
        height=30,
        rail_generator=sparse_rail_generator(
            max_num_cities=3,
            seed=123,
            grid_mode=False,
            max_rails_between_cities=2,
            max_rail_pairs_in_city=2,
        ),
        line_generator=sparse_line_generator(),
        number_of_agents=5,
    )

    obs, info = env.reset(regenerate_rail=True, regenerate_schedule=True, random_seed=123)
    del obs, info

    render_tool = RenderTool(env, gl="PGL")

    max_steps = 250
    action_choices = [
        RailEnvActions.DO_NOTHING,
        RailEnvActions.MOVE_LEFT,
        RailEnvActions.MOVE_FORWARD,
        RailEnvActions.MOVE_RIGHT,
        RailEnvActions.STOP_MOVING,
    ]

    for step in range(max_steps):
        action_dict = {
            agent_idx: random.choice(action_choices).value
            for agent_idx in range(env.get_num_agents())
        }
        _, _, done, _ = env.step(action_dict)


        # Live render in a window; this does not save video/images.
        render_tool.render_env(
            show=True,
            show_agents=True,
            show_inactive_agents=False,
            show_observations=False,
            show_predictions=False,
            frames=False,
        )

        time.sleep(0.05)

        if done.get("__all__", False):
            break

    render_tool.close_window()


if __name__ == "__main__":
    main()