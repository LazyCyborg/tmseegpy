ICLABEL_NUMERICAL_TO_STRING: dict = {
    0: "brain",
    1: "muscle artifact",
    2: "eye blink",
    3: "heart beat",
    4: "line noise",
    5: "channel noise",
    6: "other",
}

ICLABEL_STRING_TO_NUMERICAL: dict = {
    val: key for key, val in ICLABEL_NUMERICAL_TO_STRING.items()
}
