#!/usr/bin/env python3
# Converts University of Wyoming files to SHARPpy compatible format
# Created 23 May 2022 by Sam Gardner <stgardner4@tamu.edu>

import sys
from os import path, remove
import pandas as pd
from io import BytesIO


# python3 wyToSharppy.py <input> <ICAO> <%Y%m%d%H%M> <output>

if __name__ == "__main__":
    inputfilePath = sys.argv[1]
    origFile = open(inputfilePath, "r")
    origStr = origFile.read()
    origFile.close()
    origStr = origStr.replace("       ", "    NaN")
    str_bytes = BytesIO(origStr.encode())
    sound = pd.read_csv(str_bytes, delim_whitespace=True, skiprows=1).iloc[2:].dropna(how="any")
    sound = sound[["PRES", "HGHT", "TEMP", "DWPT", "DRCT", "SKNT"]].reset_index(drop=True)
    print(sound)