import ctypes
import ctypes.util

_ff = None

@ctypes.CFUNCTYPE(None, ctypes.c_int)
def custom_exit(status):
    print(f"Intercepted exit call with status: {status}")
    raise SystemExit(f"Intercepted C exit with status {status}")

def close_cdll(library):
    libsystem = ctypes.CDLL(ctypes.util.find_library("System"))
    libsystem.dlclose.argtypes = [ctypes.c_void_p]
    libsystem.dlclose.restype = ctypes.c_int
    result = libsystem.dlclose(ctypes.cast(library._handle, ctypes.c_void_p))
    if result != 0:
        raise OSError("Failed to close library handle")

def close(lib):
    close_cdll(lib)

def plan(args):
    global _ff
    _ff = ctypes.CDLL('./FF-v2.3/ff')
    _ff.main.restype = ctypes.POINTER(ctypes.c_char_p)
    _ff.main.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_char_p))
    _ff.free_memory.argtypes = [ctypes.POINTER(ctypes.c_char_p)]
    _ff.free_memory.restype = None
    _ff.exit = custom_exit
    try:
        result = _ff.main(len(args), args)
    except SystemExit as e:
        print(e)  # Handle the intercepted exit
        result = (ctypes.c_char_p * 0)
    return result

def free_memory(args):
    global _ff
    _ff.free_memory(args)
    close(_ff)