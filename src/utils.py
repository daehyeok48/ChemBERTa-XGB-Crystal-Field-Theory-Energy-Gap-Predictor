def banner(text: str):
    """
    Print a formatted banner for CLI section headers.

    This function is used to clearly separate major steps in the
    command-line interface (training, inference, etc.) and improve
    readability for the user.

    Parameters
    ----------
    text : str
        Header text to display inside the banner.
    """
    print("=" * 60)
    print(text)
    print("=" * 60)
