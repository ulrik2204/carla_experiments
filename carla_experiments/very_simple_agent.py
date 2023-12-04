from carla_experiments.carla_utils.setup import game_loop, initialize_carla


def main():
    _, world = initialize_carla()
    tasks = []
    actors = []

    game_loop(world, tasks, actors)


if __name__ == "__main__":
    main()
