import requests

# GitHub repository details
USER = "commaai"
REPO = "cereal"
BRANCH = "master"
FILES = ["car.capnp", "log.capnp", "legacy.capnp", "custom.capnp"]

# GitHub API URLs
REPO_URL = f"https://api.github.com/repos/{USER}/{REPO}"
LATEST_COMMIT_URL = f"{REPO_URL}/commits/{BRANCH}"


def download_file(url, file_name):
    response = requests.get(url)
    if response.status_code == 200:
        with open(file_name, "wb") as f:
            f.write(response.content)
        print(f"Downloaded {file_name}")
    else:
        print(f"Failed to download {file_name}: HTTP {response.status_code}")


def main():
    # Fetch the latest commit SHA
    response = requests.get(LATEST_COMMIT_URL)
    if response.status_code == 200:
        commit_sha = response.json()["sha"]
        print(f"Latest commit SHA: {commit_sha}")

        # Download files using the latest commit SHA
        for file in FILES:
            file_url = (
                f"https://raw.githubusercontent.com/{USER}/{REPO}/{commit_sha}/{file}"
            )
            download_file(file_url, "new_" + file)
    else:
        print(f"Failed to fetch latest commit: HTTP {response.status_code}")


if __name__ == "__main__":
    main()
