from openpilot_exploration.common.openpilot_repo import download_github_file

# GitHub repository details
USER = "commaai"
REPO = "cereal"
BRANCH = "master"
FILES = ["car.capnp", "log.capnp", "legacy.capnp", "custom.capnp"]

# GitHub API URLs
REPO_URL = f"https://api.github.com/repos/{USER}/{REPO}"
LATEST_COMMIT_URL = f"{REPO_URL}/commits/{BRANCH}"


def main():
    for file in FILES:
        save_path = f"carla_experiments/custom_logreader/new_{file}"
        download_github_file(
            repo_owner=USER,
            repo_name=REPO,
            file_path_in_repo=file,
            save_path=save_path,
            main_branch_name=BRANCH,
        )


if __name__ == "__main__":
    main()
