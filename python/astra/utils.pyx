# -----------------------------------------------------------------------
# Copyright: 2010-2022, imec Vision Lab, University of Antwerp
#            2013-2022, CWI, Amsterdam
#
# Contact: astra@astra-toolbox.com
# Website: http://www.astra-toolbox.com/
#
# This file is part of the ASTRA Toolbox.
#
#
# The ASTRA Toolbox is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# The ASTRA Toolbox is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with the ASTRA Toolbox. If not, see <http://www.gnu.org/licenses/>.
#
# -----------------------------------------------------------------------
#
# distutils: language = c++
# distutils: libraries = astra

import sys
cimport numpy as np
import numpy as np
import builtins
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.list cimport list
from libcpp.set cimport set
from cython.operator cimport dereference as deref, preincrement as inc

from . cimport PyXMLDocument
from .PyXMLDocument cimport XMLDocument
from .PyXMLDocument cimport XMLNode
from .PyIncludes cimport *

from .pythonutils import GPULink, checkArrayForLink
from .log import AstraError

cdef extern from "CFloat32CustomPython.h":
    cdef cppclass CDataStoragePython[T](CDataMemory[T]):
        CDataStoragePython(np.ndarray arrIn)

cdef extern from "Python.h":
    void* PyLong_AsVoidPtr(object)

cdef extern from *:
    XMLConfig* dynamic_cast_XMLConfig "dynamic_cast<astra::XMLConfig*>" (Config*)

cdef extern from "astra/Config.h" namespace "astra":
    cdef cppclass ConfigCheckData:
        set[string] parsedNodes
        set[string] parsedOptions


# TODO:
# returning configs from C++ back to Python as dicts
# PluginAlgorithm should be able to accept all types of Config

cdef cppclass PythonConfig(Config):
    dict m_dict
    dict m_options

    __init__(dict d, dict options):
        this.m_dict = d
        this.m_options = options
    bool has(const string& name) const:
        n = wrap_from_bytes(name)
        return n in m_dict
    bool hasOption(const string& name) const:
        n = wrap_from_bytes(name)
        return n in m_options

    bool getInt(const string& name, int &iValue) const:
        n = wrap_from_bytes(name)
        try:
            (&iValue)[0] = m_dict[n]
        except:
            return False
        return True

    bool getFloat(const string& name, float &fValue) const:
        n = wrap_from_bytes(name)
        try:
            (&fValue)[0] = m_dict[n]
        except:
            return False
        return True

    bool getString(const string& name, string &sValue) const:
        n = wrap_from_bytes(name)
        try:
            (&sValue)[0] = m_dict[n]
        except:
            return False
        return True

    bool getDoubleArray(const string& name, vector[double] &values) const:
        n = wrap_from_bytes(name)
        try:
            d = m_dict[n]
            if isinstance(d, np.ndarray):
                d = d.reshape(-1)
            values.clear()
            values.reserve(len(d))
            for i in d:
                values.push_back(i)
        except:
            return False
        return True

    bool getIntArray(const string& name, vector[int] &values) const:
        n = wrap_from_bytes(name)
        try:
            d = m_dict[n]
            if isinstance(d, np.ndarray):
                d = d.reshape(-1)
            values.clear()
            values.reserve(len(d))
            for i in d:
                values.push_back(i)
        except:
            return False
        return True

    bool getOptionFloat(const string& name, float &fValue) const:
        n = wrap_from_bytes(name)
        try:
            (&fValue)[0] = m_options[n]
        except:
            return False
        return True

    bool getOptionInt(const string& name, int &iValue) const:
        n = wrap_from_bytes(name)
        try:
            (&iValue)[0] = m_options[n]
        except:
            return False
        return True

    bool getOptionUInt(const string& name, unsigned int &iValue) const:
        n = wrap_from_bytes(name)
        try:
            (&iValue)[0] = m_options[n]
        except:
            return False
        return True

    bool getOptionBool(const string& name, bool &bValue) const:
        n = wrap_from_bytes(name)
        try:
            (&bValue)[0] = m_options[n]
        except:
            return False
        return True

    bool getOptionString(const string& name, string &sValue) const:
        n = wrap_from_bytes(name)
        try:
            (&sValue)[0] = m_options[n]
        except:
            return False
        return True

    bool getOptionIntArray(const string& name, vector[int] &values) const:
        n = wrap_from_bytes(name)
        try:
            d = m_options[n]
            if isinstance(d, np.ndarray):
                d = d.reshape(-1)
            values.clear()
            values.reserve(len(d))
            for i in d:
                values.push_back(i)
        except:
            return False
        return True

    bool getSubConfig(const string& name, Config *&_cfg, string& _type) const:
        n = wrap_from_bytes(name)
        try:
            d = m_dict[n]
            (&_cfg)[0] = dictToConfig(name, d)
            if 'type' in d:
                (&_type)[0] = wrap_to_bytes(d['type'])
            else:
                (&_type)[0] = b''
        except:
            return False
        return True

    list[string] checkUnparsed(const ConfigCheckData &data) const:
        errors = list[string]()

        for key in m_dict.keys():
            if key == 'type':
                # We're not monitoring the type attribute
                continue
            if key in ['option', 'Option', 'options', 'Options']:
                # TODO: This handles name clashes differently than reading
                options = m_dict[key]
                for okey in options.keys():
                    if data.parsedOptions.find(wrap_to_bytes(okey)) == data.parsedOptions.end():
                        errors.push_back(wrap_to_bytes(okey))
                continue

            if data.parsedNodes.find(wrap_to_bytes(key)) == data.parsedNodes.end():
                errors.push_back(wrap_to_bytes(key))


        return errors

    void setType(const string &tpe):
        t = wrap_from_bytes(tpe)
        m_dict["type"] = t

    void setInt(const string &name, int iValue):
        n = wrap_from_bytes(name)
        m_dict[n] = iValue

    void setDouble(const string &name, double fValue):
        n = wrap_from_bytes(name)
        m_dict[n] = fValue

    void setFloatArray(const string &name, const float *pfValues, unsigned int iCount):
        n = wrap_from_bytes(name)
        outArr = np.empty((iCount,), dtype=np.float32, order='C')
        cdef float [:] mView = outArr
        cdef const float [:] cView = <const float[:iCount]> pfValues
        mView[:] = cView
        m_dict[n] = outArr

    void setDoubleMatrix(const string &name, const vector[double] &fValues, unsigned int iHeight, unsigned int iWidth):
        n = wrap_from_bytes(name)
        outArr = np.empty((iHeight,iWidth), dtype=np.float64, order='C')
        cdef double[:,::1] mView = outArr
        cdef const double * t = &fValues[0]
        cdef const double[:,::1] cView = <const double[:iHeight, :iWidth]> t
        mView[:] = cView
        m_dict[n] = outArr

    void setOptionDouble(const string &name, double fValue):
        n = wrap_from_bytes(name)
        m_options[n] = fValue

cdef createPythonConfig(Config **_cfg):
    cdef Config *cfg
    d = { "option": {} }
    cfg = new PythonConfig(d, d["option"])
    _cfg[0] = cfg
    return d

include "config.pxi"


cdef Config * dictToConfig(string rootname, dc) except NULL:
    # TODO: Is it okay to drop the root name?
    # TODO: exception handling?
    # TODO: warn with clashing keys?

    # Do the config option parsing here to avoid potential exceptions in the
    # PythonConfig constructor.
    options = { }
    if "option" in dc:
        options.update(dc["option"])
    if "options" in dc:
        options.update(dc["options"])
    if "Option" in dc:
        options.update(dc["Option"])
    if "Options" in dc:
        options.update(dc["Options"])

    cdef PythonConfig * cfg = new PythonConfig(dc, options)
    return cfg

def convert_item(item):
    if isinstance(item, str):
        return item.encode('ascii')

    if type(item) is not dict:
        return item

    out_dict = {}
    for k in item:
        out_dict[convert_item(k)] = convert_item(item[k])
    return out_dict


def wrap_to_bytes(value):
    if isinstance(value, bytes):
        return value
    return str(value).encode('ascii')


def wrap_from_bytes(value):
    return value.decode('ascii')


cdef bool readDict(XMLNode root, _dc) except False:
    cdef XMLNode listbase
    cdef XMLNode itm
    cdef int i
    cdef int j
    cdef double* data

    dc = convert_item(_dc)
    for item in dc:
        val = dc[item]
        if isinstance(val, builtins.list) or isinstance(val, tuple):
            val = np.array(val,dtype=np.float64)
        if isinstance(val, np.ndarray):
            if val.size == 0:
                break
            listbase = root.addChildNode(item)
            contig_data = np.ascontiguousarray(val,dtype=np.float64)
            data = <double*>np.PyArray_DATA(contig_data)
            if val.ndim == 2:
                listbase.setContent(data, val.shape[1], val.shape[0], False)
            elif val.ndim == 1:
                listbase.setContent(data, val.shape[0])
            else:
                raise AstraError("Only 1 or 2 dimensions are allowed")
        elif isinstance(val, dict):
            if item == b'option' or item == b'options' or item == b'Option' or item == b'Options':
                readOptions(root, val)
            else:
                itm = root.addChildNode(item)
                readDict(itm, val)
        else:
            if item == b'type':
                root.addAttribute(< string > b'type', <string> wrap_to_bytes(val))
            else:
                if isinstance(val, builtins.bool):
                    val = int(val)
                itm = root.addChildNode(item, wrap_to_bytes(val))
    return True

cdef bool readOptions(XMLNode node, dc) except False:
    cdef XMLNode listbase
    cdef XMLNode itm
    cdef int i
    cdef int j
    cdef double* data
    for item in dc:
        val = dc[item]
        if node.hasOption(item):
            raise AstraError('Duplicate Option: %s' % item)
        if isinstance(val, builtins.list) or isinstance(val, tuple):
            val = np.array(val,dtype=np.float64)
        if isinstance(val, np.ndarray):
            if val.size == 0:
                break
            listbase = node.addChildNode(b'Option')
            listbase.addAttribute(< string > b'key', < string > item)
            contig_data = np.ascontiguousarray(val,dtype=np.float64)
            data = <double*>np.PyArray_DATA(contig_data)
            if val.ndim == 2:
                listbase.setContent(data, val.shape[1], val.shape[0], False)
            elif val.ndim == 1:
                listbase.setContent(data, val.shape[0])
            else:
                raise AstraError("Only 1 or 2 dimensions are allowed")
        else:
            if isinstance(val, builtins.bool):
                val = int(val)
            node.addOption(item, wrap_to_bytes(val))
    return True

cdef configToDict(Config *cfg):
    # TODO: Accept both XMLConfig and PythonConfig
    # (Need to add function to extract dictionary from PythonConfig)
    # Then call this function from PythonPluginAlgorithm
    cdef XMLConfig* xmlcfg;
    xmlcfg = dynamic_cast_XMLConfig(cfg);
    if not xmlcfg:
        return None
    return XMLNode2dict(xmlcfg.self)

def castString(input):
    return input.decode('utf-8')

def stringToPythonValue(inputIn):
    input = castString(inputIn)
    # matrix
    if ';' in input:
        input = input.rstrip(';')
        row_strings = input.split(';')
        col_strings = row_strings[0].split(',')
        nRows = len(row_strings)
        nCols = len(col_strings)

        out = np.empty((nRows,nCols))
        for ridx, row in enumerate(row_strings):
            col_strings = row.split(',')
            for cidx, col in enumerate(col_strings):
                out[ridx,cidx] = float(col)
        return out

    # vector
    if ',' in input:
        input = input.rstrip(',')
        items = input.split(',')
        out = np.empty(len(items))
        for idx,item in enumerate(items):
            out[idx] = float(item)
        return out

    try:
        # integer
        return int(input)
    except ValueError:
        try:
            #float
            return float(input)
        except ValueError:
            # string
            return str(input)


cdef XMLNode2dict(XMLNode node):
    cdef XMLNode subnode
    cdef list[XMLNode] nodes
    cdef list[XMLNode].iterator it
    dct = {}
    opts = {}
    if node.hasAttribute(b'type'):
        dct['type'] = castString(node.getAttribute(b'type'))
    nodes = node.getNodes()
    it = nodes.begin()
    while it != nodes.end():
        subnode = deref(it)
        if castString(subnode.getName())=="Option":
            if subnode.hasAttribute(b'value'):
                opts[castString(subnode.getAttribute(b'key'))] = stringToPythonValue(subnode.getAttribute(b'value'))
            else:
                opts[castString(subnode.getAttribute(b'key'))] = stringToPythonValue(subnode.getContent())
        else:
            dct[castString(subnode.getName())] = stringToPythonValue(subnode.getContent())
        inc(it)
    if len(opts)>0: dct['options'] = opts
    return dct

cdef CFloat32VolumeData3D* linkVolFromGeometry(CVolumeGeometry3D *pGeometry, data) except NULL:
    cdef CFloat32VolumeData3D * pDataObject3D = NULL
    cdef CDataStorage * pStorage
    geom_shape = (pGeometry.getGridSliceCount(), pGeometry.getGridRowCount(), pGeometry.getGridColCount())
    if isinstance(data, np.ndarray):
        data_shape = data.shape
    elif isinstance(data, GPULink):
        data_shape = (data.z, data.y, data.x)
    if geom_shape != data_shape:
        raise ValueError("The dimensions of the data {} do not match those "
                         "specified in the geometry {}".format(data_shape, geom_shape))

    if isinstance(data, np.ndarray):
        checkArrayForLink(data)
        if data.dtype == np.float32:
            pStorage = new CDataStoragePython[float32](data)
        else:
            raise NotImplementedError("Unknown data type for link")
    elif isinstance(data, GPULink):
        IF HAVE_CUDA==True:
            hnd = wrapHandle(<float*>PyLong_AsVoidPtr(data.ptr), data.x, data.y, data.z, data.pitch/4)
            pStorage = new CDataGPU(hnd)
        ELSE:
            raise AstraError("CUDA support is not enabled in ASTRA")
    else:
        raise TypeError("data should be a numpy.ndarray or a GPULink object")
    pDataObject3D = new CFloat32VolumeData3D(pGeometry, pStorage)
    return pDataObject3D

cdef CFloat32ProjectionData3D* linkProjFromGeometry(CProjectionGeometry3D *pGeometry, data) except NULL:
    cdef CFloat32ProjectionData3D * pDataObject3D = NULL
    cdef CDataStorage * pStorage
    geom_shape = (pGeometry.getDetectorRowCount(), pGeometry.getProjectionCount(), pGeometry.getDetectorColCount())
    if isinstance(data, np.ndarray):
        data_shape = data.shape
    elif isinstance(data, GPULink):
        data_shape = (data.z, data.y, data.x)
    if geom_shape != data_shape:
        raise ValueError("The dimensions of the data {} do not match those "
                         "specified in the geometry {}".format(data_shape, geom_shape))

    if isinstance(data, np.ndarray):
        checkArrayForLink(data)
        if data.dtype == np.float32:
            pStorage = new CDataStoragePython[float32](data)
        else:
            raise NotImplementedError("Unknown data type for link")
    elif isinstance(data, GPULink):
        IF HAVE_CUDA==True:
            hnd = wrapHandle(<float*>PyLong_AsVoidPtr(data.ptr), data.x, data.y, data.z, data.pitch/4)
            pStorage = new CDataGPU(hnd)
        ELSE:
            raise AstraError("CUDA support is not enabled in ASTRA")
    else:
        raise TypeError("data should be a numpy.ndarray or a GPULink object")
    pDataObject3D = new CFloat32ProjectionData3D(pGeometry, pStorage)
    return pDataObject3D

cdef CProjectionGeometry3D* createProjectionGeometry3D(geometry) except NULL:
    cdef Config *cfg
    cdef CProjectionGeometry3D * pGeometry

    cfg = dictToConfig(b'ProjectionGeometry', geometry)
    tpe = geometry['type']
    if (tpe == "parallel3d"):
        pGeometry = <CProjectionGeometry3D*> new CParallelProjectionGeometry3D();
    elif (tpe == "parallel3d_vec"):
        pGeometry = <CProjectionGeometry3D*> new CParallelVecProjectionGeometry3D();
    elif (tpe == "cone"):
        pGeometry = <CProjectionGeometry3D*> new CConeProjectionGeometry3D();
    elif (tpe == "cone_vec"):
        pGeometry = <CProjectionGeometry3D*> new CConeVecProjectionGeometry3D();
    else:
        raise ValueError("'{}' is not a valid 3D geometry type".format(tpe))

    if not pGeometry.initialize(cfg[0]):
        del cfg
        del pGeometry
        raise AstraError('Geometry class could not be initialized', append_log=True)

    del cfg

    return pGeometry

cdef CVolumeGeometry3D* createVolumeGeometry3D(geometry) except NULL:
    cdef Config *cfg
    cdef CVolumeGeometry3D * pGeometry
    cfg = dictToConfig(b'VolumeGeometry', geometry)
    pGeometry = new CVolumeGeometry3D()
    if not pGeometry.initialize(cfg[0]):
        del cfg
        del pGeometry
        raise AstraError('Geometry class could not be initialized', append_log=True)

    del cfg

    return pGeometry
