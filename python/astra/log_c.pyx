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

from libcpp.string cimport string
import logging

cdef extern from "astra/Logging.h" namespace "astra":
    cdef enum log_level:
        LOG_DEBUG
        LOG_INFO
        LOG_WARN
        LOG_ERROR

cdef extern from "astra/Logging.h" namespace "astra::CLogger":
    void debug(const char *sfile, int sline, const char *fmt, ...)
    void info(const char *sfile, int sline, const char *fmt, ...)
    void warn(const char *sfile, int sline, const char *fmt, ...)
    void error(const char *sfile, int sline, const char *fmt, ...)
    void setOutputCallback(void (*)(int, const string &, int, const string &) noexcept)
    void setOutputScreen(int fd, log_level m_eLevel)
    void setOutputFile(const char *filename, log_level m_eLevel)
    void enable()
    void enableScreen()
    void enableFile()
    void disable()
    void disableScreen()
    void disableFile()
    void setFormatFile(const char *fmt)
    void setFormatScreen(const char *fmt)
    string getLastErrMsg()

cdef void log_callback(int level, const string &file, int line, const string &msg) noexcept:
    strmsg = msg.decode('ascii')
    if level == LOG_DEBUG:
        logging.debug(strmsg)
    elif level == LOG_INFO:
        logging.info(strmsg)
    elif level == LOG_WARN:
        logging.warning(strmsg)
    elif level == LOG_ERROR:
        logging.error(strmsg)
    else:
        pass

setOutputCallback(log_callback)
disableScreen()

def log_debug(sfile, sline, message):
    sfile = sfile.encode('ascii')
    message = message.encode('ascii')
    debug(sfile,sline,"%s",<char*>message)

def log_info(sfile, sline, message):
    sfile = sfile.encode('ascii')
    message = message.encode('ascii')
    info(sfile,sline,"%s",<char*>message)

def log_warn(sfile, sline, message):
    sfile = sfile.encode('ascii')
    message = message.encode('ascii')
    warn(sfile,sline,"%s",<char*>message)

def log_error(sfile, sline, message):
    sfile = sfile.encode('ascii')
    message = message.encode('ascii')
    error(sfile,sline,"%s",<char*>message)

def log_enable():
    enable()

def log_enableScreen():
    enableScreen()

def log_enableFile():
    enableFile()

def log_disable():
    disable()

def log_disableScreen():
    disableScreen()

def log_disableFile():
    disableFile()

def log_setFormatFile(fmt):
    fmt = fmt.encode('ascii')
    setFormatFile(fmt)

def log_setFormatScreen(fmt):
    fmt = fmt.encode('ascii')
    setFormatScreen(fmt)

enumList = [LOG_DEBUG,LOG_INFO,LOG_WARN,LOG_ERROR]

def log_setOutputScreen(fd, level):
    setOutputScreen(fd, enumList[level])

def log_setOutputFile(filename, level):
    filename = filename.encode('ascii')
    setOutputFile(filename, enumList[level])

def log_getLastErrMsg():
    return getLastErrMsg().decode('UTF-8')
