#!/usr/bin/env python3
# Converts University of Wyoming files to SHARPpy compatible format
# Created 23 May 2022 by Sam Gardner <stgardner4@tamu.edu>

import sys
from os import path, remove
import pandas as pd
from datetime import datetime as dt


# python3 wyToSharppy.py <input> <ICAO> <%Y%m%d%H%M> <output>

if __name__ == "__main__":
    inputfilePath = sys.argv[1]
    origFile = open(inputfilePath, "r")
    origStr = origFile.read()
    origFile.close()
    origStr = origStr.replace("       ", "    NaN")
    tmpFilePath = inputfilePath + ".tmp"
    if path.exists(tmpFilePath):
        remove(tmpFilePath)
    tmpFile = open(tmpFilePath, "w")
    tmpFile.write(origStr)
    tmpFile.close()
    sound = pd.read_csv(inputfilePath+".tmp", delim_whitespace=True, skiprows=1).iloc[2:].dropna(how="any")
    remove(tmpFilePath)
    sound = sound[["PRES", "HGHT", "TEMP", "DWPT", "DRCT", "SKNT"]].reset_index(drop=True)
    print(sound)
    inputDateTime = dt.strptime(sys.argv[3], "%Y%m%d%H%M")
    sharppyHeader = "%TITLE%\n "+sys.argv[2]+"   "+inputDateTime.strftime("%Y%m%d/%H%M")[2:]+"\n\n   LEVEL       HGHT       TEMP       DWPT       WDIR       WSPD\n-------------------------------------------------------------------\n%RAW%\n"
    sharppyFooter = "\n%END%"
    