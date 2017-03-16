# This class takes a list of dictionaries as input and returns the same list with
# Variable length parameters removed
class RemoveVariableLength(object):
    def __init__(self):
        self.file_keylengths = {}
        self.to_remove = {}

    def delVar(self, data, key, subkey, subsubkey=None):
        full_key = key + '.' + subkey
        if subsubkey is not None:
            full_key += '.' + subsubkey

        if self.to_remove.get(full_key, False):
            if subsubkey is not None:
                del data[key][subkey][subsubkey]
            else:
                del data[key][subkey]

    def transform(self,data_list):
        self.find_key_lengths(data_list)
        self.find_remove_params()

        # remove from original data
        for index, data in data_list:
            for key, v in data.items():
                if isinstance(v, dict):
                    for subkey, subv in v.items():
                        if isinstance(subv, dict):
                            for subsubkey, subsubv in subv.items():
                                self.delVar(data, key, subkey, subsubkey)
                        else:
                            self.delVar(data, key, subkey, None)

        return data_list

    def find_remove_params(self):
        to_compare = {}
        #compare second file onwards with first file result
        for i, v in self.file_keylengths.items():
            for key, val in v.items():
                if key not in to_compare.keys():
                    to_compare[key] = val
                elif val != to_compare[key]:
                    self.to_remove[key] = True

    def find_key_lengths(self,data_list):

        for index, data in data_list:
            keywise_len = {}
            stack = [(k, v) for k, v in data.items()]
            while stack:
                k, v = stack.pop()
                if isinstance(v, dict):
                    stack.extend([(k + '.' + k1, v1) for k1, v1 in v.items()])
                elif isinstance(v, list):
                    stack.extend([(k + '.' + str(i), v[i]) for i in range(len(v))])
                else:
                    key_list = k.split(".")
                    full_key = ''
                    for key in key_list:
                        if not key.isdigit():
                            full_key += '.' + key if full_key else key

                    if full_key in keywise_len.keys():
                        keywise_len[full_key] += 1
                    else:
                        keywise_len[full_key] = 1

            self.file_keylengths[index] = keywise_len
