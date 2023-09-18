from sklearn import datasets


def _loading_example_dataset() -> None:
    """
    This function covers the first part of the tutorial that
    shows how data is loaded and analysed
    """

    digits = datasets.load_digits()

    print("\n")
    print(digits.data)
    print("\n")

    print(f"Target: {digits.target}")
    print(f"Shape: {digits.data.shape}")
    print("\n")

    # The images array is constrained by the row dimension
    # found from digits.data.shape.   It looks like
    # digits.target returns the size of each digits.images
    # dataset.
    print(f"Image[0]: \n{digits.images[0]}\n")

    print(f"Image[1]: \n{digits.images[1]}\n")

    print(f"Image[1796]: \n{digits.images[1796]}\n")


if __name__ == "__main__":

    _loading_example_dataset()
