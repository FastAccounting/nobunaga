# mode
MODE_BBOX = "bbox"

# error type
ERROR_TYPE_CLASS = "Cls"
ERROR_TYPE_LOCATION = "Loc"
ERROR_TYPE_BOTH = "Both"
ERROR_TYPE_DUPLICATE = "Dupe"
ERROR_TYPE_BACKGROUND = "Bkg"
ERROR_TYPE_MISS = "Miss"
MAIN_ERRORS = [
    ERROR_TYPE_CLASS,
    ERROR_TYPE_LOCATION,
    ERROR_TYPE_BOTH,
    ERROR_TYPE_DUPLICATE,
    ERROR_TYPE_BACKGROUND,
    ERROR_TYPE_MISS,
]

# special error type
ERROR_FALSE_POSITIVE = "FP"
ERROR_FALSE_NEGATIVE = "FN"
ERROR_TRUE_POSITIVE = "TP"
ERROR_TRUE_NEGATIVE = "TN"

SPECIAL_ERRORS = [
    ERROR_TRUE_POSITIVE,
    ERROR_TRUE_NEGATIVE,
    ERROR_FALSE_POSITIVE,
    ERROR_FALSE_NEGATIVE,
]

# threshold
THRESHOLD_MIN_DETECTED = 0.2