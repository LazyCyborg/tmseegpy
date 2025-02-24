# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#  This file (loader.py) is part of neurone_loader                             -
#  (https://www.github.com/heilerich/neurone_loader)                           -
#  Copyright © 2019 Felix Heilmeyer.                                           -
#                                                                              -
#  This code is released under the MIT License                                 -
#  https://opensource.org/licenses/mit-license.php                             -
#  Please see the file LICENSE for details.                                    -
# ------------------------------------------------------------------------------
"""
Provides classes to load, represent and export data recorded with the Bittium NeurOne device.
"""

import os
import pandas as pd
import numpy as np
from operator import indexOf

from . import neurone as nr
from .lazy import Lazy, preloadable
from .mne_export import MneExportable
from .util import logger, doc_inherit


# noinspection PyAbstractClass
class BaseContainer(MneExportable):
    """
    A metaclass that provides properties for accessing data shared between all subclasses. I cannot be used itself
    as it is not implementing all required methods of its abstract superclass.
    """

    def __init__(self):
        self._dropped_channels = set()

    @property
    def sampling_rate(self):
        """
        :return: the sampling rate, read from the session protocol
        :rtype: int
        """
        return self._protocol['meta']['sampling_rate'] if hasattr(self, '_protocol') else None

    @property
    def subject_info(self):
        """
        :return: subject information, read from the session protocol. Possible keys are
                 `id`, `first_name`, `last_name`, `date_of_birth`
        :rtype: dict
        """
        return self._protocol['meta']['subject'] if hasattr(self, '_protocol') else None

    @property
    def time_start(self):
        """
        :return: recording start, read from the session protocol
        :rtype: datetime
        """
        return self._time_start if hasattr(self, '_time_start') else None

    @property
    def time_stop(self):
        """
        :return: recording end, read from the session protocol
        :rtype: datetime
        """
        return self._time_stop if hasattr(self, '_time_stop') else None

    def _protocol_channels(self):
        return self._protocol['channels'] if hasattr(self, '_protocol') else []

    def _drop_indexes(self):
        return sorted([indexOf(self._protocol_channels(), channel) for channel in self._dropped_channels], reverse=True)

    @Lazy
    def channels(self):
        """
        :return: ordered list of all channel names, read from the session protocol
        :rtype: list[str]
        """
        return [channel for channel in self._protocol_channels() if channel not in self._dropped_channels]

    def _has_data(self):
        private_attribute_name = getattr(type(self), 'data').private_name
        return hasattr(self, private_attribute_name)

    def _extend_droplist(self, channels_to_drop):
        self._dropped_channels |= set(channels_to_drop)
        if type(getattr(type(self), 'channels')) is Lazy:
            if hasattr(self, getattr(type(self), 'channels').private_name):
                self.channels = [channel for channel in self.channels if channel not in self._dropped_channels]

    def drop_channels(self, channels_to_drop):
        """
        Remove specified channels from loaded data. Dropped channels will be remembered and when data is cleared from
        memory and reloaded from disk the channels will get removed again. To get them back create a new object of this
        type to reload from disk.

        :param channels_to_drop: names of channels to drop
        :type channels_to_drop: list[str]
        """
        drop_set = (set(channels_to_drop) - self._dropped_channels).intersection(set(self.channels))
        not_dropping = set(channels_to_drop) - drop_set
        if len(not_dropping) > 0:
            logger.warning('Not dropping channels {channels} since they don\'t exist or have already been dropped'
                           .format(channels=', '.join(not_dropping)))
        logger.debug('Dropping channels {channels}'.format(channels=', '.join(drop_set)))
        self._extend_droplist(drop_set)


@preloadable
class Phase(BaseContainer):
    """
    Represents one recording phase of one NeurOne session in one NeurOne Recording

    :param path: path to the recording *session* folder
    :param phase: phase object from a session protocol
    :type path: str
    :type phase: dict
    """
    def __init__(self, path, phase, protocol=None):
        BaseContainer.__init__(self)
        self.path = path
        self.number = phase['number']
        if protocol is None:
            self._protocol = nr.read_neurone_protocol(self.path)
        else:
            self._protocol = protocol
        self._time_start = phase['time_start']
        self._time_stop = phase['time_stop']

    @Lazy
    def events(self):
        """
        :return: recorded events with Revision, Type, SourcePort, ChannelNumber, Code, StartSampleIndex,
                 StopSampleIndex, DescriptionLength, DescriptionOffset, DataLength, DataOffset, StartTime, StopTime
        :rtype: pandas.DataFrame
        """
        return pd.DataFrame(nr.read_neurone_events(self.path, self.number, self.sampling_rate)['events'])
    
    @property
    def event_codes(self):
        """
        :return: all event codes used in the data as int32 in an numpy.ndarray
        :rtype: numpy.ndarray
        """
        return np.unique(self.events['Code'].values) if 'Code' in self.events else []

    @Lazy
    def data(self):
        """
        :return: recorded data with shape (samples, channels) in µV
        :rtype: numpy.ndarray
        """
        data = nr.read_neurone_data(self.path, self.number, self._protocol) / 1000  # data is nanovolts
        return np.delete(data, self._drop_indexes(), axis=1)

    @property
    def n_samples(self):
        """
        :return: the number of channels, inferred from the binary recording's file size
        :rtype: int
        """
        return nr.read_neurone_data_info(self.path, self.number, self._protocol).n_samples

    @property
    def n_channels(self):
        """
        :return: the number of channels, read from the session protocol
        :rtype: int
        """
        return nr.read_neurone_data_info(self.path,
                                         self.number,
                                         self._protocol).n_channels - len(self._dropped_channels)

    def clear_data(self):
        """
        Remove loaded data from memory
        """
        del self.data

    # noinspection PyMissingOrEmptyDocstring
    @doc_inherit
    def drop_channels(self, channels_to_drop):
        if self._has_data():
            drop_indexes = sorted([indexOf(self.channels, channel) for channel in channels_to_drop
                                   if channel in self.channels],
                                  reverse=True)  # ignore channels that were dropped before
            # noinspection PyAttributeOutsideInit
            self.data = np.delete(self.data, drop_indexes, axis=1)
        BaseContainer.drop_channels(self, channels_to_drop)


@preloadable
class Session(BaseContainer):
    """
    Represents one session in one NeurOne Recording and contains all of the session's phases

    :param path: path to the recording *session* folder
    :type path: str
    """
    def __init__(self, path):
        BaseContainer.__init__(self)
        self.path = path
        self._protocol = nr.read_neurone_protocol(self.path)
        self._get_meta()

    def _get_meta(self):
        self._time_start = self._protocol['meta']['time_start']
        self._time_stop = self._protocol['meta']['time_stop']
        assert len(self._protocol['phases']) > 0, \
            "Session at {} has no phases".format(self.path)
        self.phases = [Phase(self.path, p, self._protocol) for p in self._protocol['phases']]

    @property
    def event_codes(self):
        """
        :return: all event codes used in the data as int32 in an :class:`numpy.ndarray`
        :rtype: numpy.ndarray
        """
        return np.unique(np.concatenate([phase.event_codes for phase in self.phases]))

    @Lazy
    def data(self):
        """
        .. warning:: Calling this replaces the data attribute of the contained phases with a view on the concatenated
             data to save memory. Keep this in mind when manipulating the contained sessions.

        :return: concatenated data of all phases with shape (samples, channels) in µV
        :rtype: numpy.ndarray
        """
        phases = sorted(self.phases, key=lambda phase: phase.number)
        new_array = None
        slices = []
        for p in phases:
            p.drop_channels(list(self._dropped_channels))
            if new_array is None:
                new_array = np.copy(p.data, order='C')
                slices.append((0, len(p.data)))
            else:
                old_length = len(new_array)
                shape = list(new_array.shape)
                shape[0] += len(p.data)
                # data is explicitly copied above and following slices are
                new_array.resize(shape, refcheck=False)
                # appended hence also copied, this should be fine
                new_array[-len(p.data):] = p.data # pylint: disable=unsupported-assignment-operation
                slices.append((old_length, old_length + len(p.data)))
            del p.data

        for index, p in enumerate(phases):
            start, stop = slices[index]
            p.data = new_array[start:stop]

        return new_array

    def clear_data(self):
        """
        Remove loaded data in all phases from memory
        """
        for p in self.phases:
            p.clear_data()
        del self.data

    @property
    def events(self):
        """
        :return: concatenated events of all phases with Revision, Type, SourcePort, ChannelNumber, Code,
                 StartSampleIndex, StopSampleIndex, DescriptionLength, DescriptionOffset, DataLength, DataOffset,
                 StartTime, StopTime
        :rtype: pandas.DataFrame
        """
        phases = sorted(self.phases, key=lambda p: p.number)
        all_events = [phases[0].events]
        current_samples = phases[0].n_samples
        for i in range(1, len(phases)):
            if len(phases[i].events) > 0:
                cur_events = phases[i].events.copy()
                cur_events['StartSampleIndex'] += current_samples
                cur_events['StopSampleIndex'] += current_samples
                cur_time = int(current_samples / self.sampling_rate)
                cur_events['StartTime'] += cur_time
                cur_events['StopTime'] += cur_time
                current_samples += phases[i].n_samples
                all_events.append(cur_events)
        return pd.concat(all_events)
        
    @property
    def n_samples(self):
        """
        :return: sum of the number of samples, inferred from the binary recording's file size, of all phases
        :rtype: int
        """
        return sum([p.n_samples for p in self.phases])

    @property
    def n_channels(self):
        """
        Returns the number of channels used in all phases and makes sure they're equal

        :return: the number of channels, read from the session protocol
        :rtype: int
        """
        assert len(set([p.n_channels for p in self.phases])) <= 1, \
            "The number of channels shouldn't change between phases"
        return self.phases[0].n_channels if len(self.phases) > 0 else 0

    # noinspection PyMissingOrEmptyDocstring
    @doc_inherit
    def drop_channels(self, channels_to_drop):
        if self._has_data():
            # noinspection PyAttributeOutsideInit
            drop_indexes = sorted([indexOf(self.channels, channel) for channel in channels_to_drop
                                   if channel in self.channels],
                                  reverse=True)  # ignore channels that were dropped before
            # noinspection PyAttributeOutsideInit
            self.data = np.delete(self.data, drop_indexes, axis=1)
            p_offset = 0
            for phase in self.phases:
                phase._extend_droplist(channels_to_drop)
                data_length = phase.data.shape[0]
                phase.data = self.data[p_offset:p_offset + data_length]
                p_offset += data_length
        else:
            for phase in self.phases:
                phase.drop_channels(channels_to_drop)
        BaseContainer.drop_channels(self, channels_to_drop)


@preloadable
class Recording(BaseContainer):
    """
    Represents one NeurOne Recording and contains all of the recording's sessions

    :param path: path to the recording *recording* folder
    :type path: str
    """
    def __init__(self, path):
        BaseContainer.__init__(self)
        self.path = path
        self._find_sessions()
        
    def _find_sessions(self):
        session_dirs = [os.path.join(self.path, dirname)
                        for dirname in os.listdir(self.path)
                        if os.path.isdir(os.path.join(self.path, dirname))
                        and 'Protocol.xml' in os.listdir(os.path.join(self.path, dirname))]
        assert len(session_dirs) > 0, "No sessions found in {}".format(self.path)
        self.sessions = list(sorted([Session(path) for path in session_dirs], key=lambda s: s.time_start))
        self._time_start = self.sessions[0].time_start if len(self.sessions) > 0 else None
        self._time_stop = self.sessions[-1].time_stop if len(self.sessions) > 0 else None

    @property
    def event_codes(self):
        """
        :return: all event codes used in the data as int32 in an numpy.ndarray
        :rtype: numpy.ndarray
        """
        return np.unique(np.concatenate([session.event_codes for session in self.sessions]))

    @Lazy
    def data(self):
        """
        :return: concatenated data of all phases of all sessions with shape (samples, channels) in µV
        :rtype: numpy.ndarray

        .. warning:: Calling this replaces the data attribute of the contained phases and sessions with a view on the
                     concatenated data to save memory. Keep this in mind when manipulating the contained sessions or
                     phases.
        """
        sessions = sorted(self.sessions, key=lambda x: x.time_start)
        new_array = None
        slices = []
        all_phase_slices = []
        for s in sessions:
            old_length = len(new_array) if new_array is not None else 0
            if s._has_data():
                new_length = old_length + len(s.data)
            else:
                new_length = old_length + s.n_samples
            del s.data

            phases = sorted(s.phases, key=lambda phase: phase.number)
            phase_slices = []
            for p in phases:
                p.drop_channels(list(self._dropped_channels))
                if new_array is None:
                    new_array = np.copy(p.data, order='C')
                    phase_slices.append((0, len(p.data)))
                else:
                    old_phase_length = len(new_array)
                    shape = list(new_array.shape)
                    shape[0] += len(p.data)
                    # data is explicitly copied above and following slices are
                    new_array.resize(shape, refcheck=False)
                    # appended hence also copied, this should be fine
                    new_array[-len(p.data):] = p.data  # pylint: disable=unsupported-assignment-operation
                    phase_slices.append((old_phase_length, old_phase_length + len(p.data)))
                del p.data

            all_phase_slices.append(phase_slices)
            slices.append((old_length, new_length))

        for s_index, s in enumerate(sessions):
            for p_index, p in enumerate(s.phases):
                start, stop = all_phase_slices[s_index][p_index]
                p.data = new_array[start:stop]
            start, stop = slices[s_index]
            s.data = new_array[start:stop]

        return new_array

    def clear_data(self):
        """
        Remove loaded data in all phases of all sessions from memory
        """
        for s in self.sessions:
            s.clear_data()
        del self.data

    @property
    def events(self):
        """
        :return: concatenated events of all phases of all sessions with Revision, Type, SourcePort, ChannelNumber, Code,
                 StartSampleIndex, StopSampleIndex, DescriptionLength, DescriptionOffset, DataLength, DataOffset,
                 StartTime, StopTime
        :rtype: pandas.DataFrame
        """
        sessions = sorted(self.sessions, key=lambda x: x.time_start)
        assert len(set([s.sampling_rate for s in sessions])) >= 1, \
            'Loading Sessions with different sampling rates is not supported at this time'
        sampling_rate = sessions[0].sampling_rate
        all_events = [sessions[0].events]
        current_samples = sessions[0].n_samples
        for i in range(1, len(sessions)):
            if len(sessions[i].events) > 0:
                cur_events = sessions[i].events.copy()
                cur_events['StartSampleIndex'] += current_samples
                cur_events['StopSampleIndex'] += current_samples
                cur_time = int(current_samples / sampling_rate)
                cur_events['StartTime'] += cur_time
                cur_events['StopTime'] += cur_time
                current_samples += sessions[i].n_samples
                all_events.append(cur_events)
        return pd.concat(all_events)

    @property
    def n_samples(self):
        """
        :return: sum of the number of samples, inferred from the binary recording's file size, of all phases of all
                 sessions
        :rtype: int
        """
        return sum([s.n_samples for s in self.sessions])

    @property
    def n_channels(self):
        """
        Returns the number of channels used in all phases and makes sure they're equal

        :return: the number of channels, read from the session protocol
        :rtype: int
        """
        assert len(set([s.n_channels for s in self.sessions])) <= 1, \
            "The number of channels shouldn't change between sessions"
        return self.sessions[0].n_channels if len(self.sessions) > 0 else 0

    @property
    def sampling_rate(self):
        """
        Returns the sampling rate used in all sessions and makes sure they're all equal

        :return: the sampling rate, read from the session protocols
        :rtype: int
        """
        assert len(set([s.sampling_rate for s in self.sessions])) <= 1, \
            "The sampling rate shouldn't change between sessions"
        return self.sessions[0].sampling_rate if len(self.sessions) > 0 else 0

    @property
    def channels(self):
        """
        Returns the channels used in all sessions and makes sure they're equal

        :return: ordered list of all channel names, read from the session protocols
        :rtype: list[str]
        """
        assert len(set([''.join(s.channels) for s in self.sessions])) <= 1, \
            "Channel names shouldn't change between sessions"
        return [channel for channel in self.sessions[0].channels
                if channel not in self._dropped_channels] if len(self.sessions) > 0 else 0

    # noinspection PyMissingOrEmptyDocstring
    @doc_inherit
    def drop_channels(self, channels_to_drop):
        if self._has_data():
            # noinspection PyAttributeOutsideInit
            drop_indexes = sorted([indexOf(self.channels, channel) for channel in channels_to_drop
                                   if channel in self.channels],
                                  reverse=True)  # ignore channels that were dropped before
            # noinspection PyAttributeOutsideInit
            self.data = np.delete(self.data, drop_indexes, axis=1)
            s_offset = 0
            p_offset = 0
            # update views and droplists
            for session in self.sessions:
                session._extend_droplist(channels_to_drop)
                data_length = session.data.shape[0]
                session.data = self.data[s_offset:s_offset + data_length]
                s_offset += data_length
                for phase in session.phases:
                    phase._extend_droplist(channels_to_drop)
                    data_length = phase.data.shape[0]
                    phase.data = self.data[p_offset:p_offset + data_length]
                    p_offset += data_length
        else:
            for session in self.sessions:
                session.drop_channels(channels_to_drop)
        BaseContainer.drop_channels(self, channels_to_drop)
