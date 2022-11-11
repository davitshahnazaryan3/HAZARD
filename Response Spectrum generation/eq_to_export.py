import numpy as np


def ReadNGA(inFilename=None, content=None, outFilename=None):
    """
    Details
    -------
    This function process acceleration history for NGA data file (.AT2 format).
    Parameters
    ----------
    inFilename : str, optional
        Location and name of the input file.
        The default is None
    content    : str, optional
        Raw content of the .AT2 file.
        The default is None
    outFilename : str, optional
        location and name of the output file.
        The default is None.
    Notes
    -----
    At least one of the two variables must be defined: inFilename, content.
    Returns
    -------
    dt   : float
        time interval of recorded points.
    npts : int
        number of points in ground motion record file.
    desc : str
        Description of the earthquake (e.g., name, year, etc).
    t    : numpy.array (n x 1)
        time array, same length with npts.
    acc  : numpy.array (n x 1)
        acceleration array, same length with time unit
        usually in (g) unless stated otherwise.
    """

    try:
        # Read the file content from inFilename
        if content is None:
            with open(inFilename, 'r') as inFileID:
                content = inFileID.readlines()

        # check the first line
        temp = str(content[0]).split()
        try:  # description is in the end
            float(temp[0])  # do a test with str to float conversion, this will be ok if description is in the end.
            # Description of the record
            desc = content[-2]
            # Number of points and time step of the record
            row4Val = content[-4]
            # Acceleration values
            acc_data = content[:-4]
        except ValueError:  # description is in the beginning
            # Description of the record
            desc = content[1]
            # Number of points and time step of the record
            row4Val = content[3]
            # Acceleration values
            acc_data = content[4:]

        # Description of the record
        desc = desc.replace('\r', '')
        desc = desc.replace('\n', '')
        # Number of points and time step of the record
        if row4Val[0][0] == 'N':
            val = row4Val.split()
            if 'dt=' in row4Val:
                dt_str = 'dt='
            elif 'DT=' in row4Val:
                dt_str = 'DT='
            if 'npts=' in row4Val:
                npts_str = 'npts='
            elif 'NPTS=' in row4Val:
                npts_str = 'NPTS='
            if 'sec' in row4Val:
                sec_str = 'sec'
            elif 'SEC' in row4Val:
                sec_str = 'SEC'
            npts = int(val[(val.index(npts_str)) + 1].rstrip(','))
            try:
                dt = float(val[(val.index(dt_str)) + 1])
            except ValueError:
                dt = float(val[(val.index(dt_str)) + 1].replace(sec_str + ',', ''))
        else:
            val = row4Val.split()
            npts = int(val[0])
            dt = float(val[1])

        # Acceleration values
        acc = np.array([])
        for line in acc_data:
            acc = np.append(acc, np.array(line.split(), dtype=float))
        dur = len(acc) * dt
        t = np.arange(0, dur, dt)

        if outFilename is not None:
            np.savetxt(outFilename, acc, fmt='%1.4e')

        return dt, npts, desc, t, acc

    except BaseException as error:
        print(f"Record file reader FAILED for {inFilename}: ", error)
        