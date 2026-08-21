import sys
import win32com.client as win32
from tkinter import Tk, filedialog

# -----------------------------
# Helper: file chooser dialog
# -----------------------------
def choose_excel_file():
    root = Tk()
    root.withdraw()
    return filedialog.askopenfilename(
        title="Select Excel file",
        filetypes=[
            ("Excel files", "*.xlsx *.xls *.xlsm *.xlsb"),
        ]
    )

# -----------------------------
# Helper: convert #RRGGBB to Excel RGB int
# -----------------------------
def hex_to_rgb_int(hex_str: str) -> int:
    hex_str = hex_str.lstrip("#")
    r = int(hex_str[0:2], 16)
    g = int(hex_str[2:4], 16)
    b = int(hex_str[4:6], 16)
    return b << 16 | g << 8 | r

# -----------------------------
# Color map
# -----------------------------
COLOR_MAP = {
    "Best": hex_to_rgb_int("#C0504D"),
    "NACF": hex_to_rgb_int("#4F81BD"),
    "NCF": hex_to_rgb_int("#4F81BD"),
    "INCF": hex_to_rgb_int("#4F81BD"),
    "IXCL": hex_to_rgb_int("#4F81BD"),
    "INPCF": hex_to_rgb_int("#4F81BD"),
    "Vanilla": hex_to_rgb_int("#C0504D"),
    "XCF": hex_to_rgb_int("#C0504D"),
    "Worst": hex_to_rgb_int("#C0504D"),

    "3 variates": hex_to_rgb_int("#7F7F7F"),
    "4 variates": hex_to_rgb_int("#4EA72E"),
    "5 variates": hex_to_rgb_int("#E97132"),

    "50K": hex_to_rgb_int("#156082"),
    "100K": hex_to_rgb_int("#196B24"),
    "200K": hex_to_rgb_int("#A02B93"),
    "400K": hex_to_rgb_int("#0D3A4E"),

    "Positive": hex_to_rgb_int("#BDBDFF"),
    "Mixed": hex_to_rgb_int("#A4EA58"),
    "Negative": hex_to_rgb_int("#96CEFC"),

    "Wide": hex_to_rgb_int("#0F9ED5"),
    "Narrow": hex_to_rgb_int("#009E73"),

    "Baseline": hex_to_rgb_int("#DE82A5"),
    "Lexsort": hex_to_rgb_int("#8FC3A3"),
}

# -----------------------------
# Patterns
# -----------------------------
PATTERN_LIST = [
    None,  # no pattern
    34,    # msoPatternLargeGrid
    36,    # msoPatternLargeCheckerBoard
    14,    # msoPatternDarkVertical
    13,    # msoPatternDarkHorizontal
    33,    # msoPatternLargeConfetti
    32,    # msoPatternDashedHorizontal
    25,    # msoPatternWideDownwardDiagonal
    26,    # msoPatternWideUpwardDiagonal
]

PATTERN_MAP = {
    "Best": PATTERN_LIST[0],
    "NACF": PATTERN_LIST[0],
    "NCF": PATTERN_LIST[0],
    "INCF": PATTERN_LIST[0],
    "IXCL": PATTERN_LIST[0],
    "INPCF": PATTERN_LIST[0],
    "Vanilla": PATTERN_LIST[0],
    "XCF": PATTERN_LIST[0],
    "Worst": PATTERN_LIST[0],

    "3 variates": PATTERN_LIST[0],
    "4 variates": PATTERN_LIST[0],
    "5 variates": PATTERN_LIST[0],

    "50K": PATTERN_LIST[0],
    "100K": PATTERN_LIST[0],
    "200K": PATTERN_LIST[0],
    "400K": PATTERN_LIST[0],

    "Positive": PATTERN_LIST[0],
    "Mixed": PATTERN_LIST[0],
    "Negative": PATTERN_LIST[0],

    "Wide": PATTERN_LIST[0],
    "Narrow": PATTERN_LIST[0],

    "Baseline": PATTERN_LIST[0],
    "Lexsort": PATTERN_LIST[0],
}

# Column/bar chart types
COLUMN_BAR_TYPES = [51, 52, 53, 57, 58, 59]

# -----------------------------
# Open Excel
# -----------------------------
excel = win32.gencache.EnsureDispatch("Excel.Application")
excel.Visible = False  # optional

file_path = choose_excel_file()
if not file_path:
    sys.exit(0)

wb = excel.Workbooks.Open(file_path)

# -----------------------------
# Apply per-category color + pattern (single series)
# -----------------------------
for ws in wb.Sheets:
    for chart_obj in ws.ChartObjects():
        chart = chart_obj.Chart
        if chart.ChartType not in COLUMN_BAR_TYPES:
            continue

        # Only one series
        series = chart.SeriesCollection(1)
        categories = series.XValues

        # Handle the last point separately
        name = categories[-1]
        if name in COLOR_MAP:
            # Set fill color
            series.Format.Fill.ForeColor.RGB = COLOR_MAP[name]

            # Patterned fill if defined, otherwise solid
            if name in PATTERN_MAP and PATTERN_MAP[name] is not None:
                series.Format.Fill.Patterned(PATTERN_MAP[name])
            else:
                series.Format.Fill.Solid()
            
            # Border same as fill
            series.Format.Line.Visible = True
            series.Format.Line.ForeColor.RGB = COLOR_MAP[name]
            series.Format.Line.Weight = 1
        
        # Handle the rest of the points
        for i, point in enumerate(list(series.Points())[:-1]):
            name = categories[i]
            if name in COLOR_MAP:
                # Set fill color
                point.Format.Fill.ForeColor.RGB = COLOR_MAP[name]

                # Patterned fill if defined, otherwise solid
                if name in PATTERN_MAP and PATTERN_MAP[name] is not None:
                    point.Format.Fill.Patterned(PATTERN_MAP[name])
                else:
                    point.Format.Fill.Solid()

                # Border same as fill
                point.Format.Line.Visible = True
                point.Format.Line.ForeColor.RGB = COLOR_MAP[name]
                point.Format.Line.Weight = 1
        
        # Handle legends
        if chart.HasLegend:
            chart.Legend.Format.TextFrame2.TextRange.Font.Size = 27
            chart.Legend.Format.TextFrame2.TextRange.Font.Superscript = True
            chart.Legend.Format.TextFrame2.TextRange.Font.BaselineOffset = 0.04

# -----------------------------
# Save and close
# -----------------------------
wb.Save()
wb.Close()
excel.Quit()

print("Bar charts updated with per-category colors and patterns!")
