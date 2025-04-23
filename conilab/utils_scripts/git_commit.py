import os
import argparse


def arguments() -> dict:
    base_parser = argparse.ArgumentParser(
        prog="git script",
    )
    base_parser.add_argument(
        "-m",
        "--commit_message",
        help="Commit message",
        dest="commit_message",
        required=True,
    )
    return vars(base_parser.parse_args())


def main():
    args = arguments()
    os.system(f"git add {os.getcwd()}")
    os.system(f"""git commit -m "{args["commit_message"]}" """)
    os.system("git push")


if __name__ == "__main__":
    main()
