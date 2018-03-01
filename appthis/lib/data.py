import json
import tarfile as tf


def data_iterator(tarfile):
    with tf.open(tarfile) as data_tarfile:
        for tarfile_member in data_tarfile:
            fname = tarfile_member.name
            if fname.endswith('.dat') and tarfile_member.isfile():
                data_file = data_tarfile.extractfile(tarfile_member)
                for line in data_file:
                    line = line.decode(encoding='UTF-8')
                    line = json.loads(line)
                    yield line
                data_file.close()
