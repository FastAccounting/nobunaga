def print_table(rows: list, title: str = None):
    print()
    # Get all rows to have the same number of columns
    max_cols = max([len(row) for row in rows])
    for row in rows:
        while len(row) < max_cols:
            row.append("")

    # Compute the text width of each column
    col_widths = [
        max([len(rows[i][col_idx]) for i in range(len(rows))]) for col_idx in range(len(rows[0]))
    ]

    divider = "--" + ("---".join(["-" * w for w in col_widths])) + "-"
    thick_divider = divider.replace("-", "=")

    if title:
        left_pad = (len(divider) - len(title)) // 2
        print(("{:>%ds}" % (left_pad + len(title))).format(title))

    print(thick_divider)
    for row in rows:
        # Print each row while padding to each column's text width
        print(
            "  "
            + "   ".join(
                [
                    ("{:>%ds}" % col_widths[col_idx]).format(row[col_idx])
                    for col_idx in range(len(row))
                ]
            )
            + "  "
        )
        if row == rows[0]:
            print(divider)

    print(divider)

    # output sum row
    column_sum = [0 for _ in range(len(col_widths))]
    column_sum[0] = "Sum"
    for row in rows[1:]:
        for row_index, row_value in enumerate(row):
            if row_index == 0:
                continue
            column_sum[row_index] = int(column_sum[row_index]) + int(row_value)
    column_sum = [str(value) for value in column_sum]
    print(
        "  "
        + "   ".join(
            [
                ("{:>%ds}" % col_widths[col_idx]).format(column_sum[col_idx]) for col_idx in range(len(column_sum))
            ]
        )
        + "  "
    )
    print(thick_divider)
    print()
