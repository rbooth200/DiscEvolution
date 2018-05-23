import errno    
import os


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def make_ASCII_header(HDF5_attributes):
    """Generates header in ASCII format from the HDF5 format."""
    # Class name
    head = "# {}".format(HDF5_attributes[0])

    # Add dictionary of attributes
    def make_item(item):
        key, value = item
        return "{}: {}".format(key, value)

    return head + ", ".join(map(make_item, HDF5_attributes[1].items()))
