from conilab.data_functions.image_handling import threshold_img
import argparse
import sys
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
        default=0,
        help="Threshold value",
        type=int,
    )
    option.add_argument(
        "-T",
        "--two_sided",
        dest="two_sided",
        help="Threshold two sides",
        action="store_false",
    )

    option.add_argument(
        "-P",
        "--remove_pos_onlys",
        dest="remove_pos_onlys",
        help="Remove only positive values",
        action="store_false",
    )

    if len(sys.argv) == 1:
        option.print_help(sys.stderr)
        sys.exit(1)

    return vars(option.parse_args())


def main() -> None:
    """
    Main function

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    arg = args()
    print(f"Threshlding image at {arg['threshold']}")
    threshold_img(
        arg["image"],
        os.path.join(arg["output_dir"], arg["image"]),
        arg["threshold"],
        arg["two_sided"],
        arg["remove_pos_onlys"],
    )
    print(f"Print Saving {arg['image']} to: {arg['output_dir']}")


if __name__ == "__main__":
    main()
