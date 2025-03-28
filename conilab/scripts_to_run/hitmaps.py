import argparse
import sys
from conilab.nfact.coverage import coverage_map
import os


def args() -> dict:
    """
    Function to get arguments for hitmaps

    Parameters
    ----------
    None

    Returns
    -------
    dict: dictionary
       dictionary of args
    """
    option = argparse.ArgumentParser()
    option.add_argument(
        "-n",
        "--name_of_image",
        dest="name_of_image",
        help="""Image name to save as """,
        required=True,
    )
    option.add_argument(
        "-o",
        "--output_dir",
        dest="output_dir",
        default=os.getcwd(),
        help="""Output directory to save image as""",
    )
    option.add_argument(
        "-i", "--image", dest="image", help="Abosulte path to image", required=True
    )
    option.add_argument(
        "-t",
        "--threshold",
        dest="threshold",
        default=2,
        help="Threshold value",
        type=int,
    )
    option.add_argument(
        "-D",
        "--dont_normalise",
        dest="dont_normalise",
        help="Don't normalise and threshold image",
        action="store_false",
    )

    if len(sys.argv) == 1:
        option.print_help(sys.stderr)
        sys.exit(1)

    return vars(option.parse_args())


def main():
    """
    Main function
    """
    arg = args()
    threshold_string = (
        "Not thresholding"
        if not arg["dont_normalise"]
        else f"Thresholding at {arg['threshold']}"
    )
    print(f"Saving hitmaps to {arg['output_dir']}. {threshold_string}")
    try:
        coverage_map(
            arg["image"],
            os.path.join(arg["output_dir"], arg["name_of_image"]),
            arg["threshold"],
            arg["dont_normalise"],
        )
    except Exception as e:
        print(f"Unable to save images due to {e}")


if __name__ == "__main__":
    main()
