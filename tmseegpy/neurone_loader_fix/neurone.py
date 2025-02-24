# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#  This file (neurone.py) is part of neurone_loader                            -
#  (https://www.github.com/heilerich/neurone_loader)                           -
#  Copyright © 2019 Felix Heilmeyer.                                           -
#                                                                              -
#  This code is released under the MIT License                                 -
#  https://opensource.org/licenses/mit-license.php                             -
#  Please see the file LICENSE for details.                                    -
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# This file contains code originally from export2hdf5                          -
# (https://github.com/bwrc/export2hdf5)                                        -
# Created by Andreas Henelius <andreas.henelius@ttl.fi>,                       -
# Finnish Institute of Occupational Health                                     -
# ------------------------------------------------------------------------------

"""
Contains functions for reading data recorded with a
Bittium NeurOne device. This module currently supports
reading of data and events.
"""

import numpy as np
import xml.etree.ElementTree
import re

from os import path
from construct import Struct, Int32sl, Int64ul

from datetime import datetime

from collections import namedtuple


def read_neurone_protocol(fpath):
    """
    Read the measurement protocol from an XML file.

    Arguments:
       - fpath: the path to the directory holding the
                 NeurOne measurement (i.e., the
                 directory Protocol.xml and Session.xml
                 files.

    Returns:
       - a dictionary containing (i) the names of the channels
         in the recording and (ii) meta information
         (recording start/stop times, sampling rate).

    {"meta" : <dict with metadata>,
    "channels" : <array with channel names>}
    """

    # Define filename
    fname_protocol = path.join(fpath, "Protocol.xml")
    fname_session = path.join(fpath, "Session.xml")

    # --------------------------------------------------
    # Read the protocol data
    # --------------------------------------------------
    # Define the XML namespace as a shorthand
    ns = {'xmlns': 'http://www.megaemg.com/DataSetGeneralProtocol.xsd'}

    # Get channel names and organise them according to their
    # physical order (InputNumber), which is the order
    # in which the channels are being sampled.
    doc_root = xml.etree.ElementTree.parse(fname_protocol).getroot()
    channels = doc_root.findall("xmlns:TableInput", namespaces=ns)
    channel_names = [(0, 0)] * len(channels)

    for i, ch in enumerate(channels):
        channel_names[i] = (int(ch.findall("xmlns:PhysicalInputNumber", namespaces=ns)[0].text),
                            ch.findall("xmlns:Name", namespaces=ns)[0].text)
    channel_names = [i for _, i in sorted(channel_names)]

    # Get the sampling rate
    sampling_rate = int(doc_root.findall("xmlns:TableProtocol", namespaces=ns)[0]
                        .findall("xmlns:ActualSamplingFrequency", namespaces=ns)[0].text)

    # --------------------------------------------------
    # Read the session data
    # --------------------------------------------------
    # Define the XML namespace as a shorthand
    ns2 = {'xmlns': 'http://www.megaemg.com/DataSetGeneralSession.xsd'}

    # Get channel names and organise them according to their
    # physical order (InputNumber), which is the order
    # in which the channels are being sampled.
    doc_root = xml.etree.ElementTree.parse(fname_session).getroot()
    session = doc_root.findall("xmlns:TableSession", namespaces=ns2)
    time_start = session[0].findall("xmlns:StartDateTime", namespaces=ns2)[0].text
    time_stop = session[0].findall("xmlns:StopDateTime", namespaces=ns2)[0].text
    phases = [{'number': phase.findall("xmlns:Folder", namespaces=ns2)[0].text.split("\\")[-1],
               'time_start': phase.findall("xmlns:StartDateTime", namespaces=ns2)[0].text,
               'time_stop': phase.findall("xmlns:StopDateTime", namespaces=ns2)[0].text}
              for phase in doc_root.findall("xmlns:TableSessionPhase", namespaces=ns2)]

    subject_info = doc_root.find('xmlns:TablePerson', namespaces=ns2)
    subject_id = subject_info.find('xmlns:PersonID', namespaces=ns2).text
    subject_first_name = subject_info.find('xmlns:FirstName', namespaces=ns2).text
    subject_last_name = subject_info.find('xmlns:LastName', namespaces=ns2).text
    subject_dob = subject_info.find('xmlns:DateOfBirth', namespaces=ns2).text

    # --------------------------------------------------
    # Package the information
    # --------------------------------------------------
    meta = {}

    meta["time_start"] = _convert_time(time_start)
    meta["time_stop"] = _convert_time(time_stop)
    meta["sampling_rate"] = sampling_rate

    meta["subject"] = dict(
        id=subject_id,
        first_name=subject_first_name,
        last_name=subject_last_name,
        date_of_birth=_convert_time(subject_dob)
    )
    
    for phase in phases:
        phase['time_start'] = _convert_time(phase['time_start'])
        phase['time_stop'] = _convert_time(phase['time_stop'])

    return {'channels': channel_names, 'meta': meta, 'phases': phases}

from datetime import datetime
import re

def _convert_time(inp_str):
    """
    Converts a datetime string to a datetime object.
    Handles both naive and timezone-aware datetime strings.
    """
    try:
        # Handle 'Z' timezone
        if inp_str.endswith('Z'):
            # Replace 'Z' with '+00:00' for UTC
            inp_str = inp_str[:-1] + '+00:00'
        
        # Use regex to parse the datetime components
        iso_format_regex = r'^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(\.\d+)?([+-]\d{2}:\d{2})?$'
        match = re.match(iso_format_regex, inp_str)
        if match:
            main_time, fractional, timezone = match.groups()
            if fractional:
                # Remove the dot
                fractional = fractional[1:]
                # Limit to 6 digits by truncating or rounding
                fractional = fractional[:6].ljust(6, '0')
                microsecond = int(fractional)
            else:
                microsecond = 0
            # Reconstruct datetime string without fractional seconds
            dt_str = main_time
            if timezone:
                dt_str += timezone
            dt = datetime.fromisoformat(dt_str)
            dt = dt.replace(microsecond=microsecond)
            return dt
        else:
            raise ValueError("Input string does not match expected format.")
    except Exception as e:
        raise ValueError(f"Error parsing time string '{inp_str}': {e}")


def read_neurone_data(fpath, session_phase=1, protocol=None):
    """
    Read the NeurOne signal data from a binary file.

    Arguments:
       - fpath: the path to the directory holding the
                 NeurOne measurement (i.e., the
                 directory Protocol.xml and Session.xml
                 files.

       - session_phase:
                 The phase of the measurement. Currently
                 only reading of the first phase (1) is
                 supported.

       - protocol:
                  The dictionary obtained using the function
                  read_neurone_protocol. This argument is optional
                  and if not given, the protocol is automatically read.
                    
    Returns:
       - A numpy ndarray with the data, where each columns stores
         the data for one channel.
    """

    fname = path.join(fpath, str(session_phase), '1.bin')

    # Read the protocol unless provided
    if protocol is None:
        protocol = read_neurone_protocol(fpath)
    
    # Determine number of samples to read
    n_samples, n_channels = read_neurone_data_info(fpath, session_phase, protocol)

    # Read the data and store the data
    # in an ndarray
    data = np.fromfile(fname, dtype='<i4')
    data.shape = (n_samples, n_channels)

    return data


def read_neurone_data_info(fpath, session_phase=1, protocol=None):
    """
    Read the sample and channel count from a NeurOne signal binary file.

    Arguments:
       - fpath: the path to the directory holding the
                 NeurOne measurement (i.e., the
                 directory Protocol.xml and Session.xml
                 files.

       - session_phase:
                 The phase of the measurement. Currently
                 only reading of the first phase (1) is
                 supported.

       - protocol:
                  The dictionary obtained using the function
                  read_neurone_protocol. This argument is optional
                  and if not given, the protocol is automatically read.
                    
    Returns:
       Returns:
       - a named tuple containing (i) the number of channels
         and (ii) the number of samples in the recording.

        ( n_samples, n_channels )
    """

    fname = path.join(fpath, str(session_phase), '1.bin')

    # Read the protocol unless provided
    if protocol is None:
        protocol = read_neurone_protocol(fpath)
    
    # Determine number of samples and channels
    f_info = path.getsize(fname)
    n_channels = len(protocol['channels'])
    n_samples = int(f_info / 4 / n_channels)
    
    DataInfo = namedtuple('DataInfo', ['n_samples', 'n_channels'])
    return DataInfo(n_samples, n_channels)


def get_n1_event_format():
    """
    Define the format for the events in a neurone recording.
    
    Arguments: None.

    Returns:
       - A Struct (from the construct library) describing the
         event format.
    """

    # Define the data format of the events
    # noinspection PyUnresolvedReferences
    return Struct(
        "Revision" / Int32sl,
        "RFU1" / Int32sl,
        "Type" / Int32sl,
        "SourcePort" / Int32sl,
        "ChannelNumber" / Int32sl,
        "Code" / Int32sl,
        "StartSampleIndex" / Int64ul,
        "StopSampleIndex" / Int64ul,
        "DescriptionLength" / Int64ul,
        "DescriptionOffset" / Int64ul,
        "DataLength" / Int64ul,
        "DataOffset" / Int64ul,
        "RFU2" / Int32sl,
        "RFU3" / Int32sl,
        "RFU4" / Int32sl,
        "RFU5" / Int32sl
    )


def read_neurone_events(fpath, session_phase=1, sampling_rate=None):
    """
    Read the NeurOne events from a binary file.

    Arguments:
       - fpath: the path to the directory holding the
                 NeurOne measurement (i.e., the
                 directory Protocol.xml and Session.xml
                 files.

       - sampling_rate:
                 The sampling rate of the recording.
                 This argument is optional and if not given,
                 the protocol is automatically read.

       - session_phase:
                 The phase of the measurement. Currently
                 only reading of the first phase (1) is
                 supported.

    Returns:
       - A dict containing the events and the data type for the events.
    {"events" : <numpy structured array with the events>,
    "events_dtype" : <array with the numpy dtype for the events>}
    """

    fname = path.join(fpath, str(session_phase), "events.bin")

    # Get the sampling rate unless provided
    if sampling_rate is None:
        protocol = read_neurone_protocol(fpath)
        sampling_rate = protocol['meta']['sampling_rate']
    
    # Determine number of events
    f_info = path.getsize(fname)
    n_events = int(f_info / 88)
    events = [{}] * n_events

    # Read events in chunks of 88 bytes and unpack
    # also add start / stop time for each event
    # and remove 'reserved for future use' (RFU) fields
    event_format = get_n1_event_format()
    with open(fname, mode='rb') as file:
        for i in range(n_events):
            events[i] = event_format.parse(file.read(88))
            events[i]['StartTime'] = events[i]['StartSampleIndex'] / sampling_rate
            events[i]['StopTime'] = events[i]['StopSampleIndex'] / sampling_rate
            for j in range(5):
                del events[i]['RFU' + str(j+1)]
            del events[i]['_io']

    # Create a numpy structured array from the events
    events_dtype = np.dtype([("Revision", np.int32),
                             ("Type", np.int32),
                             ("SourcePort", np.int32),
                             ("ChannelNumber", np.int32),
                             ("Code", np.int32),
                             ("StartSampleIndex", np.int64),
                             ("StopSampleIndex", np.int64),
                             ("DescriptionLength", np.int64),
                             ("DescriptionOffset", np.int64),
                             ("DataLength", np.int64),
                             ("DataOffset", np.int64),
                             ("StartTime", np.int64),
                             ("StopTime", np.int64)])

    # convert array of event dicts to an array of tuples
    if len(events) == 0:
        return {'events': np.array([], dtype=events_dtype), 'dtype': events_dtype}
    key_list = [k for k, v in events[0].items()]
    tmp = [tuple([e[k] for k in key_list]) for e in events]
    events = np.array(tmp, dtype=events_dtype)

    return {'events': events, 'dtype': events_dtype}
