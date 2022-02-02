"""
This module provides a collection of simple functions to fetch data from the
W7-X ArchiveDB and MDSplus.
Imported from PCIanalysis package, commit b5f6e359c1169148ada4a470ab5d71acf37d9fa4
Creator: Lukas Boettger (lukb@ipp.mpg.de)
Maintainer: Adrian von Stechow (astechow@ipp.mpg.de)
3/2019 D. Smith: copied from git.ipp-hgw.mpg.de/astechow/w7x-overviewplot
"""
#from PCIanalysis import vmec
import vmec
from scipy.interpolate import interp1d
from urllib.parse import urlencode

#import PCIanalysis.process as p
import MDSplus
import requests
import numpy as np
import re
import os.path as osp
import time

# avoid an annoying issue which produces UserWarnings from uvlib in jupyter
# see https://github.com/gevent/gevent/issues/1347 for details
import warnings
warnings.simplefilter("ignore", category=UserWarning)

try:
    import grequests
    GREQ_AVAIL = True
except Exception:
    GREQ_AVAIL = False


server_url = 'http://archive-webapi.ipp-hgw.mpg.de'
archive_url = server_url + '/ArchiveDB'
signalfile = osp.join(osp.dirname(__file__), 'archive_signal_dict.txt')
with open(signalfile, 'r') as inf:
    archive_signal_dict = eval(inf.read())
thomson_chlist_OP12a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
thomson_chlist_OP12b = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                        1004, 1005, 1006, 1016, 1017, 1018, 1025, 1026, 1035,
                        1036, 1041, 1042, 1045, 1046, 1053, 1054, 1057, 1058,
                        1063, 1064, 1069, 1070, 1073, 1074, 1075, 1076]

if not GREQ_AVAIL:
    print('consider installing grequests for python for parallel async download!')


def query_archive(url, params=None):
    """lowest level function to get data from archive.
    does no version checking or anything fancy.

    Args:
        url (str): full URL to resource
        params (dict): http request parameters

    Returns:
        data (str): json response from server
    """
    response = requests.get(url, params, headers={
                            'Accept': 'application/json'})
    if not response.ok:
        raise Exception('Request failed (url: {:s}).'.format(response.url))
    data = response.json()
    if type(data) == dict:
        data['url'] = response.url
    return data


def get_version_info(url, params=None):
    """gets version info of stream.

    Args:
        url (str or list of str): full archive URL(s), with our without trailing slash
        params (dict): dict containing from and upto times.
            Defaults to None, which corresponds to all times

    Returns:
        is_versioned (bool or list of bool): True if versioning is on
        latest_version (int or list of int): latest version number

    stolen and adapted from Birger Buttenschön"""

    if type(url) is not list:
        listinput = False
        url = [url]
    else:
        listinput = True

    # URL formating
    for idx, u in enumerate(url):
        if u[-1] == '/':
            url[idx] = u + '_versions.json'
        else:
            url[idx] = u + '/_versions.json'

    if GREQ_AVAIL:
        # map requests to async threads
        reqs = (grequests.get(u, params=params) for u in url)
        out = grequests.map(reqs)
        responses = []
        # gather responses
        for r in out:
            j = r.json()
            responses.append(j)
    else:
        # no grequests available
        responses = [query_archive(u, params) for u in url]

    # prepare output
    is_versioned = [False] * len(url)
    latest_version = [-1] * len(url)

    for idx, r in enumerate(responses):
        if not r['versionInfo'] == []:
            is_versioned[idx] = True
            latest_version[idx] = r['versionInfo'][-1]['number']

    if listinput:
        return is_versioned, latest_version
    else:
        return is_versioned[0], latest_version[0]


def get_latest_version_url(url, params=None):
    """generates URL for latest version of stream.
    requires substring "/V1/" for detection

    Args:
        url (str or list of str): full archive URL (or list of URLs)
        params (dict): dict containing from and upto times.
            Defaults to None, which corresponds to all times.

    Returns:
        url (str or list of str): full archive URL(s)
    """
    if type(url) is not list:
        listinput = False
        url = [url]
    else:
        listinput = True

    # prepare URL list for version checking
    check = [True] * len(url)
    for idx, u in enumerate(url):
        if u.find('/V1/') == -1:
            check[idx] = False

    checkurls = list(np.array(url)[check])
    baseurls = [u.split('/V1/')[0] for u in checkurls]

    if len(baseurls) > 0:
        # do actual version checking with list of URLs
        versioned, latest = get_version_info(baseurls, params)

    for idx, u in enumerate(checkurls):
        if versioned[idx]:
            val = u.replace('/V1/', '/V{}/'.format(latest[idx]))
            checkurls[idx] = val

    # prepare output list
    url_out = url
    i = 0
    for idx, c in enumerate(check):
        if c:
            url_out[idx] = checkurls[i]
            i += 1

    if listinput:
        return list(url_out)
    else:
        return url_out[0]


def get_archive_byurl(url, shot, nsamples=None):
    """Simple access to archive signal by relative url.

    Args:
        url (str): relative URL of signal (e.g. '/ArchiveDB/ ... /')
        shot (int): MDSplus style shot number
        nsamples (int, optional): number of samples to get.
            Defaults to None (all samples)

    Returns:
        values (list): signal values
        time array (list): timestamps in ns relative to (first) t1
        units (str): units description provided by Archive
    """
    info = get_program_info(shot_to_program_nr(shot))
    fullurl = server_url + url + '_signal.json'
    params = {'from': info['from'],
              'upto': info['upto']}
    fullurl = get_latest_version_url(fullurl, params)
    if nsamples:
        params['reduction'] = 'minmax'
        params['nSamples'] = nsamples
    response = query_archive(fullurl,
                             params=params)
    time = np.array(response.get('dimensions')) - get_t1(shot)
    return response.get('values'), time, response.get('unit')


def get_archive_byname(name, shot, nsamples=None):
    """Simple access to archive signal by name in archive_signal_dict

    Args:
        name (str): name of signal in archive_signal_dict
        shot (int): MDSplus style shot number
        nsamples (int, optional): number of samples to get.
            Defaults to None (all samples)

    Returns:
        values (list): signal values
        time array (list): timestamps in ns relative to (first) t1
        units (str): units description provided by Archive
    """
    return get_archive_byurl(archive_signal_dict.get(name), shot, nsamples)


def get_final_url_byname(name, shot, nsamples=None):
    """Return the final URL of named signal

    Includes detected last version and all required params. Call just like get_archive_byname()

    Args:
        name (str): name of signal in archive_signal_dict
        shot (int): MDSplus style shot number
        nsamples (int, optional): number of samples to get.
            Defaults to None (all samples)

    Returns:
        url (str): full archive URL to final signal
    """
    info = get_program_info(shot_to_program_nr(shot))
    url = archive_signal_dict.get(name)
    fullurl = server_url + url + '_signal.json'
    params = {'from': info['from'],
              'upto': info['upto']}
    fullurl = get_latest_version_url(fullurl, params)
    if nsamples:
        params['reduction'] = 'minmax'
        params['nSamples'] = nsamples
    return fullurl + '?' + urlencode(params)


def get_multiple_byname(names, shot, nsamples=None):
    """get multiple named signals in multiple requests.
    Depending on the availability of grequests package, this will do
    async requests to dramatically speed up downloading.

    Args:
        names (list of str): names of signals in archive_signal_dict
        shot (int): MDSplus style shot number
        nsamples (int, optional): number of samples to download per signal

    Returns:
        data(list of tuples): list of signal tuples
            first tuple element: signal values
            second tuple lement: timestamps in ns relative to (first) t1
    """
    info = get_program_info(shot_to_program_nr(shot))
    # prepare url
    urls = [archive_signal_dict.get(name) for name in names]
    fullurls = [server_url + url + '_signal.json/' for url in urls]
    params = {'from': info['from'],
              'upto': info['upto']}
    if nsamples is not None:
        params['reduction'] = 'minmax'
        params['nSamples'] = nsamples
    fullurls = get_latest_version_url(fullurls, params=params)
    t1 = get_t1(shot)
    if GREQ_AVAIL:
        return _get_multiple_grequests(fullurls, params, t1)
    else:
        return _get_multiple_serialrequests(fullurls, params, t1)


def _get_multiple_serialrequests(fullurls, params, offset):
    """get multiple signals by archive URL in serial requests.
    Advantage over get_archive_byurl: Only gets t1 once.

    Args:
        fullurls (list of str): list of full URLs to fetch
        params (list of dicts OR dict): dict of query params
        offset (int): time of T1 trigger in ns

    Returns:
        data (list of lists): data in format [[data, dim], [data, dim], ... ]
    """
    data = []
    for idx, url in enumerate(fullurls):
        if type(params) == list:
            param = params[idx]
        else:
            param = params
        try:
            d = query_archive(url, params=param)
            data.append((d['values'], np.array(d['dimensions']) - offset))
        except Exception:
            data.append((np.array([]), np.array([])))
    return data


def _get_multiple_grequests(fullurls, params, offset):
    """fetch multiple signals by archive URL in parallel requests.
    Uses grequests package.

    Args:
        fullurls (list of str): list of full URLs to fetch
        params (list of dicts OR dict): dict of query params
        offset (int): time of T1 trigger in ns

    Returns:
        data (list of lists): data in format [[data, dim], [data, dim], ... ]
    """
    if type(params) == list:
        reqs = (grequests.get(tup[0], params=tup[1])
                for tup in zip(fullurls, params))
    else:
        reqs = (grequests.get(url, params=params) for url in fullurls)
    results = grequests.map(reqs)
    data = []
    for r in results:
        j = r.json()
        data.append((j['values'], np.array(j['dimensions']) - offset))
    return data


def check_availability(name, shot):
    """returns intervals for named signal in which data is available

    Args:
        name (str): name of signal in archive_signal_dict
        shot: MDSplus style shot number

    Returns:
        times (list of tuples): start and end times of
            segments in ns (w7x timestamp) returned from archive.
    """
    url = archive_signal_dict.get(name)
    info = get_program_info(shot_to_program_nr(shot))
    fullurl = server_url + url[:-1]
    try:
        response = query_archive(fullurl,
                                 params={'filterstart': info['from'],
                                         'filterstop': info['upto']})
    except Exception:
        return []

    links = response['_links']['children']
    segments = []
    for link in links:
        start = int(re.search('from=([0-9]+)&upto', link['href']).group(1))
        end = int(re.search('&upto=([0-9]+)', link['href']).group(1))
        segments.append((start, end))
    return segments


def get_thomson_all(shot):
    """get thomson scattering profiles: n_e, T_e, ne_err, Te_err

    Args:
        shot (int): MDSplus style shot number

    Returns:
        thomsondata (dict): dictionary of:
            'ne': density
            'Te': temperature
            'ne_err': density error estimate
            'Te_err': temperature error estimate
            'reff': effective radius of channels for shot
            't': time array
            'chlist': array of scattering volume numbers
    """

    # get data in single call to grequests for best async calls
    # only a little faster than just calling get_thomson_ne() etc.
    prefixlist = [
        'ne_ts_', 'ne_ts_hi_', 'ne_ts_lo_',
        'Te_ts_', 'Te_ts_hi_', 'Te_ts_lo_']
    data, chlist, t = _get_thomson(shot, prefixlist)
    n = int(len(chlist) / 6)

    # get reff of scattering volumes, don't use field line tracer
    reff, _ = get_thomson_reff(shot, noflt=True)

    thomsondata = {
        'ne': data[:, 0 * n:1 * n],
        'ne_err':  data[:, 1 * n:2 * n] - data[:, 2 * n:3 * n],
        'Te': data[:, 3 * n:4 * n],
        'Te_err':  data[:, 4 * n:5 * n] - data[:, 5 * n:6 * n],
        'reff': reff,
        't': t,
        'chlist': chlist[:n]
    }
    return thomsondata


def get_thomson_ne(shot):
    """get thomson scattering ne profiles

    Args:
        shot (int): MDSplus style shot number

    Returns:
        ne (2D array): 2D numpy array with density in 1e19 m^3
            first dimension: time
            second dimenion: channel
        chlist (1D array): list of thomson scattering channels
        t (1D array): time in ns
    """
    return _get_thomson(shot, 'ne_ts_')


def get_thomson_ne_err(shot):
    """get thomson scattering ne profile errors

    Args:
        shot (int): MDSplus style shot number

    Returns:
        ne (2D array): 2D numpy array with density in 1e19 m^3
            first dimension: time
            second dimenion: channel
        chlist (1D array): list of thomson scattering channels
        t (1D array): time in ns
    """
    hi, _, _ = _get_thomson(shot, 'ne_ts_hi_')
    lo, chlist, t = _get_thomson(shot, 'ne_ts_lo_')
    return hi - lo, chlist, t


def get_thomson_Te(shot):
    """get thomson scattering Te profiles

    Args:
        shot (int): MDSplus style shot number

    Returns:
        Te (2D array): 2D numpy array with Te in keV
            first dimension: time
            second dimenion: channel
        chlist (1D array): list of thomson scattering channels
        t (1D array): time in ns
    """
    return _get_thomson(shot, 'Te_ts_')


def get_thomson_Te_err(shot):
    """get thomson scattering ne profile errors

    Args:
        shot (int): MDSplus style shot number

    Returns:
        ne (2D array): 2D numpy array with density in 1e19 m^3
            first dimension: time
            second dimenion: channel
        chlist (1D array): list of thomson scattering channels
        t (1D array): time in ns

    Screwed up capitalization, this just calls get_thomson_te_err()
    """
    return get_thomson_te_err(shot)


def get_thomson_te_err(shot):
    """get thomson scattering ne profile errors

    Args:
        shot (int): MDSplus style shot number

    Returns:
        ne (2D array): 2D numpy array with density in 1e19 m^3
            first dimension: time
            second dimenion: channel
        chlist (1D array): list of thomson scattering channels
        t (1D array): time in ns
    """
    hi, _, _ = _get_thomson(shot, 'Te_ts_hi_')
    lo, chlist, t = _get_thomson(shot, 'Te_ts_lo_')
    return hi - lo, chlist, t


def _get_thomson_volumes(shot):
    """get thomson scattering volume positions

    Args:
        shot (int): MDSplus style shot number

    Returns:
        pos (2D array): 3xN array of x, y, z positions
        chlist (1D array): list of thomson scattering channels
    """
    prefixes = ['thomson_pos_x_', 'thomson_pos_y_', 'thomson_pos_z_']
    data, chlist, _ = _get_thomson(shot, prefixes)
    n = int(len(chlist) / 3)
    x = data[0][0 * n:1 * n]
    y = data[0][1 * n:2 * n]
    z = data[0][2 * n:3 * n]
    pos = np.stack([x, y, z])
    return pos, chlist[:n]


def _get_thomson(shot, prefix):
    """subfunction to get thomson data for all channels

    Args:
        shot (int): MDSplus style shot number
        prefix (str or list of str): 'ne_ts_' or 'Te_ts_' or similar
    """

    # list of available channels changed as channels were added
    if shot > 180101000:
        chlist = thomson_chlist_OP12b
    else:
        chlist = thomson_chlist_OP12a

    if type(prefix) is list:
        names = [pre + str(ch) for pre in prefix for ch in chlist]
        chlist = chlist * len(prefix)
    else:
        names = [prefix + str(ch) for ch in chlist]
    d = get_multiple_byname(names, shot)
    valid = [n for n, di in enumerate(d) if di[1].size > 0]
    chlist = [chlist[i] for i in valid]
    data = np.empty([len(d[1][0]), len(valid)])
    for counter, i in enumerate(valid):
        data[:, counter] = d[i][0]
    t = d[1][1]
    return data, chlist, t


def get_thomson_reff(shot, noflt=False):
    """gets reff values for Thomson scattering volumes.

    Args:
        shot (int): MDSplus style shot number
        noflt (bool, optional): True if field line tracer should not be run.
            Defaults to False.

    Returns:
        reff (list of float): reff values in m for channels
        chlist (list of int): list of channels
    """

    # don't need coil currents if not using field line tracer
    if noflt:
        currents = None
    else:
        currents = get_coilcurrents(shot)

    try:
        vmecids, _, _ = get_archive_byname('VMEC_id', shot)
        # vmecids is list of strings like ['w7x_ref_123'], isolate number
        confid = int(vmecids[0][8:])
    except Exception:
        print('get_thomson_reff: could not load VMEC ID from Minerva stream')
        if currents is None:
            # load coil currents for vmec threeletter code generator
            currents = get_coilcurrents(shot)
        threeletter = vmec.get_threeletter(currents)
        confid = vmec.get_confid(threeletter)

    pos, chlist = _get_thomson_volumes(shot)
    reff = vmec.get_reff(pos, currents, confid)

    # extrapolate outside of LCFS if field line tracer off (much faster)
    if noflt:
        reff = np.array(reff, dtype=float)
        reff[reff is None] = np.nan
        # find central channel
        reff_minidx = np.nanargmin(reff)
        hfschannels = pos[1] - pos[1][1] < 0
        R = np.sqrt((pos[0] - pos[0][reff_minidx]) ** 2 +
                    (pos[1] - pos[1][reff_minidx]) ** 2 +
                    (pos[2] - pos[2][reff_minidx]) ** 2)
        R[hfschannels] *= -1

        # LFS channel fit
        limval = 0.1  # fit away from funky core channels
        idx = (R > limval) & np.isfinite(reff)
        # 3rd degree polynomial fit
        fit = np.polyfit(R[idx], reff[idx], 3)
        p = np.poly1d(fit)
        # indexes of R to extrapolate to
        interpidx = (R > 0) & np.isnan(reff)
        reff[interpidx] = p(R[interpidx])

        # HFS channel fit
        limval = -0.1
        idx = (R < limval) & np.isfinite(reff)
        fit = np.polyfit(R[idx], reff[idx], 3)
        p = np.poly1d(fit)
        interpidx = (R < 0) & np.isnan(reff)
        reff[interpidx] = p(R[interpidx])

    return reff, chlist


def _get_xics_tree(shotorprgnr):
    shot = convert_prgid(shotorprgnr, 'shotnum')
    MDSplus.setenv('qsw_eval_path', 'mds-data-1.ipp-hgw.mpg.de::/w7x/eval/~t')
    return MDSplus.Tree('QSW_EVAL', shot)


def _get_xics_data(node):
    return {'vals': node.data(), 'unit': node.units,
            't_vals': node.TIME.data() / 1000, 't_unit': node.TIME.units,
            'rho_vals': node.RHO.data(), 'rho_unit': node.RHO.units,
            'reff_vals': node.REFF.data(), 'reff_unit': node.REFF.units,
            'sigma_vals': node.SIGMA.data(), 'sigma_unit': node.SIGMA.units}


def get_xics_Ti(shotorprgnr):
    """
    Get time resolved Ti profile from XICS.

    Args:
        shotorprgnr (str or int): shotnum or program number

    Returns:
        dict:
            'vals' (2D array): Ti values
            'unit' (str): unit of Ti values
            't_vals' (array): time values relativ to T1
            't_unit' (str): unit of time values
            'rho_vals' (array): rho values
            'rho_unit' (str): unit of rho values
            'reff_vals' (array): reff values
            'reff_unit' (str): unit of reff values
            'sigma_vals' (array): sigma values
            'sigma_unit' (str): unit of sigma values
    """

    tree = _get_xics_tree(shotorprgnr)
    return _get_xics_data(tree.XICS.TI)


def get_xics_Te(shotorprgnr):
    """
    Get time resolved Te profile from XICS.
    Note: Values might not be very reliable. Better use Thomson data.

    Args:
        shotorprgnr (str or int): shotnum or program number

    Returns:
        dict:
            'vals' (2D array): Te values
            'unit' (str): unit of Te values
            't_vals' (array): time values relativ to T1
            't_unit' (str): unit of time values
            'rho_vals' (array): rho values
            'rho_unit' (str): unit of rho values
            'reff_vals' (array): reff values
            'reff_unit' (str): unit of reff values
            'sigma_vals' (array): sigma values
            'sigma_unit' (str): unit of sigma values
    """

    tree = _get_xics_tree(shotorprgnr)
    return _get_xics_data(tree.XICS.TE)


def get_xics_Er(shotorprgnr):
    """
    Get time resolved Er profile from XICS.
    Note:
        * you can trust the shape, but be careful interpreting the magnitude (only approximate values)
        * for reversed field: wrong sign
        * the position of the crossover (where the sign of Er flips) is pretty robust

    Args:
        shotorprgnr (str or int): shotnum or program number

    Returns:
        dict:
            'vals' (2D array): Er values
            'unit' (str): unit of Er values
            't_vals' (array): time values relativ to T1
            't_unit' (str): unit of time values
            'rho_vals' (array): rho values
            'rho_unit' (str): unit of rho values
            'reff_vals' (array): reff values
            'reff_unit' (str): unit of reff values
            'sigma_vals' (array): sigma values
            'sigma_unit' (str): unit of sigma values
    """

    tree = _get_xics_tree(shotorprgnr)
    return _get_xics_data(tree.XICS.ER)


def get_xics_velocity(shotorprgnr):
    """
    Get time resolved velocity profile from XICS.
    Note: see get_xics_Er(...) for details.

    Args:
        shotorprgnr (str or int): shotnum or program number

    Returns:
        dict:
            'vals' (2D array): velocity values
            'unit' (str): unit of velocity values
            't_vals' (array): time values relativ to T1
            't_unit' (str): unit of time values
            'rho_vals' (array): rho values
            'rho_unit' (str): unit of rho values
            'reff_vals' (array): reff values
            'reff_unit' (str): unit of reff values
            'sigma_vals' (array): sigma values
            'sigma_unit' (str): unit of sigma values
    """

    tree = _get_xics_tree(shotorprgnr)
    return _get_xics_data(tree.XICS.VELOCITY)


def get_xics_profile_lineint(shotorprgnr):
    """get 2D line integrated (non-inverted) XICS data.

    Args:
        shotorprgnr (str or int): shotnum or program number

    Returns:
        Ti (2D array): 2D numpy array with Ti in keV
            first dimension: channel
            second dimension: time
        channels (1D array): list of channel (pixel) numbers
        t (1D array): time in s past T1
    """
    tree = _get_xics_tree(shotorprgnr)
    node = tree.XICS_LINE.TI
    ti = node.data()
    channels = node.CHANNEL.data()
    t = node.TIME.data()
    return ti, channels, t


def get_xics_center_lineint(shotorprgnr, returnpath=False):
    """get central line integrated (non-inverted) XICS data.

    Args:
        shotorprgnr (str or int): shotnum or program number
        returnpath (bool): adds an additional output argument with MDSplus tree path
            Defaults to False

    Returns:
        Ti (1D array): 1D numpy array with Ti in keV
        t (1D array): time in s past T1
        (treepath (str): exists only if returnpath=True, contains full tree path)
    """
    tree = _get_xics_tree(shotorprgnr)
    node = tree.XICS_LINE.TI0
    ti = node.data()[0]
    t = node.TIME.data()
    if returnpath:
        return ti, t, node.record
    else:
        return ti, t


def get_bremsstrahlung(shot, returnpath=False):
    """get Bremsstrahlung from filterscope data

    Args:
        shot (int): MDSplus style shot number
        returnpath (bool): adds an additional output argument with MDSplus tree path
            Defaults to False

    Returns:
        bremsstrahlung (1D array): 1D numpy array with bremsstrahlung
        time (1D array): time in s past T1
        (treepath (str): exists only if returnpath=True, contains full tree path)
    """
    MDSplus.setenv('qsr02_path', 'mds-data-2::/w7x/new/~t;mds-data-1::/w7x/vault/~t')
    tree = MDSplus.Tree('QSR02', shot)
    node = tree.HARDWARE.ARRAY.TUBE48
    data = node.data()
    # dim_of is too long (software bug on diagnostic side), cut off
    time = np.asanyarray(node.dim_of())[0:len(data)]

    if returnpath:
        return data, time, node.path
    else:
        return data, time


def get_halpha(shot, returnpath=False):
    """get H-alpha from filterscope data

    Args:
        shot (int): MDSplus style shot number
        returnpath (bool): adds an additional output argument with MDSplus tree path
            Defaults to False

    Returns:
        h-alpha (1D array): 1D numpy array with h-alpha light
        time (1D array): time in s past T1
        (treepath (str): exists only if returnpath=True, contains full tree path)
    """
    MDSplus.setenv('qsr02_path', 'mds-data-2::/w7x/new/~t;mds-data-1::/w7x/vault/~t')
    tree = MDSplus.Tree('QSR02', shot)
    node = tree.HARDWARE.ARRAY.TUBE32
    data = node.data()
    # dim_of is too long (software bug on diagnostic side), cut off
    time = np.asanyarray(node.dim_of())[0:len(data)]

    if returnpath:
        return data, time, node.path
    else:
        return data, time


def get_last_t0():
    """get most recent t0 issued

    Args:
        None

    Returns:
        time (int): w7x time of last t0 in ns
    """
    url = 'http://archive-webapi.ipp-hgw.mpg.de/last_trigger'
    return int(requests.get(url).text)


def get_t1(shot):
    """get first t1 time point of shot in ns

    Args:
        shot (int): MDSplus style shot number

    Returns:
        time (int): w7x time of shot in ns
    """
    info = get_program_info(shot_to_program_nr(shot))
    return info['trigger']['1'][0]


def get_program_of_t0(timestamp):
    """get the program ID corresponding to a t0 time stamp.
    Also works with a program that is still running!

    Args:
        timestamp (int): w7x timestamp

    Returns:
        program id (str)
    """
    params = {'from': timestamp - int(1e9) - 1,
              'upto': timestamp - int(1e9)}
    url = '/ArchiveDB/raw/W7X/ProjectDesc.1/ProgramLabelStart/parms/ProgramLogLabel/progId/'
    fullurl = server_url + url + '_signal.json'
    progid = query_archive(fullurl, params=params)
    url = '/ArchiveDB/raw/W7X/ProjectDesc.1/ProgramLabelStart/parms/ProgramLogLabel/date/'
    fullurl = server_url + url + '_signal.json'
    date = query_archive(fullurl, params=params)
    return '{}.{:03}'.format(date['values'][0], progid['values'][0])


def get_t4(shot):
    """get last t4 time point of shot in ns

    Args:
        shot (int): MDSplus style shot number

    Returns:
        time (int): w7x time of shot in ns
    """
    info = get_program_info(shot_to_program_nr(shot))
    return info['trigger']['4'][-1]


def now_in_ns():
    return int(time.time() * 1e9)


def program_nr_to_shot(program_nr):
    program_nr = str(program_nr)
    return int(program_nr[2:].replace('.', ''))


def shot_to_program_nr(shot):
    shot = str(shot)
    return '20' + shot[:6] + '.' + shot[6:]


def convert_prgid(prgid, target):
    """
    Converts prgid (either shot or program_nr) into:
        shotnum if target is "shotnum"
        program_nr if target is "program_nr"
    """

    hasdot = '.' in str(prgid)

    if target == 'shotnum':
        return program_nr_to_shot(prgid) if hasdot else int(prgid)

    if target == 'program_nr':
        return str(prgid) if hasdot else shot_to_program_nr(prgid)

    raise KeyError("'{}' is not a valid value for 'target'. Use 'shotnum' or "
                   "program_nr.".format(target))


def get_program_info(program_nr, upto=None):
    """
    Returns w7x program info.

    Args:
        program_nr (str or int): program number (like '20171207.6') or w7x ns timestamp
        upto: required only if using timestamp for program_nr

    Returns:
        programinfo (dict): keys are:
            ['from', 'upto', 'description', 'scenarios', 'id', 'sessionInfo', 'name', 'trigger'].
    """
    if upto:
        params = {'from': program_nr,
                  'upto': upto}
    else:
        params = {'from': program_nr}
    res = query_archive(server_url + '/programs.json', params)
    return res['programs'][0]


def get_PCI_trigger(program_nr):
    """
    Returns the trigger for the PCI diagnostic. All values are in nanoseconds.
    """
    tree = MDSplus.Tree('QOC', program_nr_to_shot(program_nr))
    # delay in s or ns for earlier shots, ms later
    delayNode = tree.HARDWARE.RPTRIG.DELAY
    factor = {'ns': 1,
              'ms': 1e6,
              's': 1e9}
    unit = delayNode.units
    delay = delayNode.data() * factor.get(str(unit))
    timing = np.asarray(tree.HARDWARE.RPTRIG.TIMING.data()) * 1e6
    T0_mdsplus = int(tree.getNode('TIMING.T0').data()[0])
    info = get_program_info(program_nr)
    T0 = info['trigger']['0'][0]  # T0 in ns

    if T0 != T0_mdsplus:
        raise Warning('T0 in archive and MDSplus are not equal (XP {:s})! '
                      'MDSplus shot and program number not in sync?'.format(program_nr))

    return {'T0': T0,
            'calibration start': T0 + delay,
            'calibration stop': T0 + delay + timing[1],
            'measurement start': T0 + delay + timing[2],
            'measurement stop': T0 + delay + timing[3]}


def get_ECRH(start, end):
    """
    Returns total ECRH power for given time interval [start, end] in nanoseconds.
    Returned dict contains time trace under 'dimensions' key.
    """
    diag_url = ('/raw/W7X/CoDaStationDesc.18774/'
                'FeedBackProcessDesc.18770_DATASTREAM/0/ECRH%20Total%20Power/')
    request_url = archive_url + diag_url + '_signal.json'
    res = query_archive(request_url, params={'from': start, 'upto': end})
    res['dimensions'] = np.asarray(res['dimensions'])
    res['values'] = np.asarray(res['values'])
    return res


def get_NBI(shotnum, nsamples=None):
    """Returns the total NBI power

    Args:
        shotnum (int): MDSplus-style shot number
        nsamples (int): number of samples
            Defaults to None, corresponds to length of NBI source 7 time trace

    Returns:
        power (list of float): Power in MW
        time (list of float): time in s past t1
    """

    getsamples = nsamples if nsamples is None else nsamples * 2

    U7, timeU7, _ = get_archive_byname(
        'NBI_U7', shotnum, nsamples=getsamples)
    U8, timeU8, _ = get_archive_byname(
        'NBI_U8', shotnum, nsamples=getsamples)
    I7, timeI7, _ = get_archive_byname(
        'NBI_I7', shotnum, nsamples=getsamples)
    I8, timeI8, _ = get_archive_byname(
        'NBI_I8', shotnum, nsamples=getsamples)
    status, _, _ = get_archive_byname(
        'NBI_w7X', shotnum)

    # interpolation time trace
    if nsamples is None:
        nsamples = len(U7)
    ti = np.linspace(-0.2, timeU7[-1], nsamples)

    # power calculation as given by NBI group
    P = 0.35 * (np.interp(ti, timeU7, U7, left=0, right=0) / 10 * 73 *
                np.interp(ti, timeI7, I7, left=0, right=0) / 10 * 0.25 +
                np.interp(ti, timeU8, U8, left=0, right=0) / 10 * 73 *
                np.interp(ti, timeI8, I8, left=0, right=0) / 10 * 0.25
                )

    if not any(np.asarray(status) > 0):
        P = []

    return P, ti / 1e9


def get_ECRH_positions(shotnum):
    """Returns the ECRH launching position for all gyrotrons.

    Args:
        shot (int): MDSplus style shot number

    Returns:
        echlabels (list of str): labels of ECH launchers
        zoffs (list of float): vertical offset in mm, None if not available
        phioffs (list of float): toroidal offset in degrees, None if not available

    """
    echlabels = ['A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'A5', 'B5', 'C5', 'D5', 'E5', 'F5']
    zofflabels = ['ECRH_{}_zoff'.format(echlabel) for echlabel in echlabels]
    phiofflabels = ['ECRH_{}_phioff'.format(echlabel) for echlabel in echlabels]
    d = get_multiple_byname(zofflabels, shotnum)
    zoffs = [di[0][0] if len(di[0]) > 0 else None for di in d]
    d = get_multiple_byname(phiofflabels, shotnum)
    phioffs = [di[0][0] if len(di[0]) > 0 else None for di in d]

    return echlabels, zoffs, phioffs


def get_ECRH_setpoint(shotnum, nsamples=5000):

    active, t, _ = get_archive_byname('ECRH_active', shotnum)
    signames = ['ECRH_setpoint_' + a + b for a, b in zip(active[0][::2], active[0][1::2])]
    data = get_multiple_byname(signames, shotnum, nsamples)
    shotlen = (get_t4(shotnum) - get_t1(shotnum)) / 1e9
    ti = np.linspace(0, shotlen, nsamples)
    di = np.zeros(len(ti))
    for trace in data:
        try:
            t = trace[1] / 1e9
            d = trace[0]
            di = di + np.interp(ti, t, d, left=0, right=0)
        except Exception:
            continue

    return di, ti


def get_mean_HHeratio(shotnum):
    """Returns the mean Hydrogen/Helium ratio from the edge

    Args:
        shot (int): MDSplus style shot number
    """
    d, _, _ = get_archive_byname('HHeratio', shotnum)
    d = np.array(d)
    return np.mean(np.compress(d > 0.02, d))


def get_coilcurrents(shotnum):
    coils = ['NPC1', 'NPC2', 'NPC3', 'NPC4', 'NPC5', 'PCA', 'PCB']
    currents = []
    d = get_multiple_byname(coils, shotnum)
    currents = [di[0][0] for di in d]

    return currents


def get_gasflows(shotnum, nsamples=None):
    """Returns the total gas flow from standard piezo valves BGXXX

    shotnum (int): MDSplus-style shot number
    nsamples (int, optional): number of samples to get.
        Defaults to None (all samples)
    """

    # gas valve voltage signal names
    valve_volt_names = ['ValveV{}'.format(i) for i in range(0, 11)]

    # get gas valve voltage traces
    valve_voltages = []
    d = get_multiple_byname(valve_volt_names, shotnum, nsamples)
    valve_voltages = [di[0] for di in d]

    # find traces with values > 10V
    ids = np.where(np.any(np.asarray(valve_voltages) > 10, 1))[0]

    # flow signal names
    valve_flow_names = ['ValveF{}'.format(i) for i in ids]

    # get gas flow traces
    flows = []
    d = get_multiple_byname(valve_flow_names, shotnum, nsamples)
    flows = [di[0] for di in d]
    time = d[0][1]

    return np.asarray(flows).T, time, ids


def get_gastypes(shotnum, ids=None):

    # get all valves if not otherwise specified
    if ids is None:
        ids = range(0, 11)

    # get gas type
    gastype_names = ['ValveGas{}'.format(i) for i in ids]

    # get gas flow traces
    gastypenums = []
    d = get_multiple_byname(gastype_names, shotnum)
    gastypenums = [di[0][0] for di in d]

    gaslist = {1: 'no bottle',
               2: 'undefined',
               3: 'evacuated',
               4: 'res1',
               5: 'res2',
               6: 'B2H6',
               7: 'SiH4',
               8: 'He',
               9: 'D2',
               10: 'H2',
               11: 'N2',
               12: 'Ne',
               13: 'Ar',
               14: 'Kr',
               15: 'CH4',
               }
    gastypes = [gaslist[i] for i in gastypenums]

    return gastypes


def get_gas_setpoints(shotnum, ids=None, nsamples=None):

    # get all valves if not otherwise specified
    if ids is None:
        ids = range(0, 11)

    # setpoint signal names
    setpoint_names = ['ValveSP{}'.format(i) for i in ids]

    # get gas setpoint traces
    setpoints = []
    d = get_multiple_byname(setpoint_names, shotnum, nsamples)
    setpoints = [di[0] for di in d]
    time = d[0][1]

    return np.asarray(setpoints).T, time


def get_interlock(shotnum):
    """gets interlock times and reasons

    Args:
        shotnum (int): MDSplus-style shot number

    Returns:
        interlocked (bool): True if any interlock
        times (list of float): interlock times in s after T1
        reasons (list of str): reasons for specific interlocks
    """
    interlock_names = ['interlock_QXD', 'interlock_QMJ_ECRH',
                       'interlock_ECRH']
    # removed: 'interlock_QMJ_NBI'
    interlock_reasons = ['Wdia low', 'density low', 'ECH sniffer']
    # removed: density low'

    d = get_multiple_byname(interlock_names, shotnum)
    data = np.asarray([di[0] for di in d])
    time = d[0][1] / 1e9

    interlocked = np.any(data[:])
    interlocks = np.argwhere(np.diff(data, axis=1) > 0)
    tind = interlocks[:, 1]
    systemind = interlocks[:, 0]

    if interlocked:
        reasons = [interlock_reasons[i] for i in systemind]
        times = time[tind]
    else:
        reasons = ['no interlock']
        times = [None]

    return interlocked, times, reasons


def densityscaler(rawsignal, shotnum):
    """Returns rescaled density signal.
    Use for old interferometry signal before correct calibration was applied.

    Args:
        rawsignal (array): raw interferometer signal
        shotnum (int): MDSplus-style shot number

    Returns:
        line integrated density (array) in m^-2
    """
    # constants from https://w7x-logbook/components?id=QMJ
    if type(rawsignal) == float:
        return np.nan
    if shotnum < 170914000:
        offset = -9989.0
        scaling = -11179218513509570.0
    elif shotnum < 170927000:
        offset = -26386.0
        scaling = -2227794915994841.0
    elif shotnum < 171005000:
        offset = -26386.0
        scaling = -6683384747984523.0
    elif shotnum < 171018000:
        offset = -19577.0
        scaling = +2279257185939055.5
    elif shotnum > 171018000:
        offset = -np.mean(rawsignal[0:200])
        scaling = +2279257185939055.5
    else:
        raise Exception('no density scaling known for shot')
    return scaling * (rawsignal + offset)


def get_tauE(shotorprgnr, dtsmooth=50e-3, smooth='average', Pc=1500, fulldata=False):
    """
    Calculates the energy confinement time tau_E.

    Args:
        shotorprgnr (str or int): shotnum or program number
        dtsmooth (float): timescale in s for data smoothing
        smooth (str or fct): function used for smoothing:
            'average': a running mean is used
            'lowpass': lowpass filter is used
            <lambda>: arbitray function with signature fct(values, dtsmooth, fs),
            where 'values' is the 1D data array to be smootehd and fs (float)
            the sampling frequency with unit [1/dtsmooth].
        Pc (float): critical ECRH power in kW. tauE is only calculated for P>Pc
        fulldata (bool): If True intermediate results are returned as well.

    Returns:
        dict:
            't' (array): time trace in ns relativ to T1
            'tauE' (array): energy confinement time in seconds

            if fulldata is True additional values are returned (see also Args):

            'raw_Wdiavals' (array): used raw values of Wdia in kJ
            'raw_Wdiats' (array): corresponding time trace in ns relativ to T1
            'raw_Pvals' (array): used raw values of ECRH power in kW
            'raw_Pts' (array): corresponding time trace in ns relativ to T1
            'Wdia' (array): smoothed Wdia in kJ, corresponding time trace is 't'
            'dWdia' (array): smoothed dWdia/dt in kW, corresponding time trace is 't'
            'P' (array): smoothed ECRH power in kW, corresponding time trace is 't'

    """
    shot = convert_prgid(shotorprgnr, 'shotnum')
    kind = 'linear'
    bounds_error = False
    res = {}

#    if smooth == 'lowpass':
#        def lowpass(values, dt, fs):
#            return p.lowpass(values, 1 / dt, fs)
#
#        smooth = lowpass
#
    if smooth == 'average':
        def running_average(values, dt, fs):
            w = np.ones(int(np.round(dt * fs)))
            return np.convolve(values, w / np.sum(w), mode='same')

        smooth = running_average

    Wdiavals, Wdiats, _ = get_archive_byname('Wdia', shot)
    Wdiavals = np.abs(Wdiavals)

    if fulldata:
        res['raw_Wdiavals'] = Wdiavals
        res['raw_Wdiats'] = Wdiats

    Wdiadt = (Wdiats[1] - Wdiats[0]) * 1e-9
    Wdiavals = smooth(Wdiavals, dtsmooth, 1 / Wdiadt)

    dWdiats = Wdiats[:-1]
    dWdiavals = (Wdiavals[1:] - Wdiavals[:-1]) / Wdiadt
    idWdiavals = interp1d(dWdiats, dWdiavals, kind=kind, bounds_error=bounds_error)

    Pvals, Pts, _ = get_archive_byname('ECRH_tot', shot)
    Pdt = (Pts[1] - Pts[0]) * 1e-9

    if fulldata:
        res['raw_Pvals'] = Pvals
        res['raw_Pts'] = Pts

    Pvals = smooth(Pvals, dtsmooth, 1 / Pdt)
    iPvals = interp1d(Pts, Pvals, kind=kind, bounds_error=bounds_error)

    new_Pvals = iPvals(Wdiats)
    new_dWdiavals = idWdiavals(Wdiats)

    tauEs = Wdiavals / (new_Pvals - new_dWdiavals)

    m = new_Pvals > Pc

    if fulldata:
        res['Wdia'] = Wdiavals[m]
        res['dWdia'] = new_dWdiavals[m]
        res['P'] = new_Pvals[m]

    res['t'] = Wdiats[m]
    res['tauE'] = tauEs[m]

    return res
