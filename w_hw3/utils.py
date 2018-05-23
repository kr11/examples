import pickle


def list_save(content, filename, mode='w'):
    if type(content) is not list:
        raise TypeError("expect list")
    file = open(filename, mode)
    for i in range(len(content)):
        file.write(str(content[i]) + '\n')
    file.close()


def list_load(filename):
    try:
        file = open(filename, 'r')
    except IOError:
        error = []
        return error
    content = file.readlines()
    for i in range(len(content)):
        content[i] = content[i][:len(content[i]) - 1]
    file.close()
    return content


def dict_save(content, filename, mode='wb'):
    if type(content) is not dict:
        raise TypeError("expect dict")
    f1 = open(filename, mode)
    pickle.dump(content, f1)
    f1.close()


def dict_load(filename):
    f2 = open(filename, 'rb')
    content = pickle.load(f2)
    f2.close()
    return content


if __name__ == '__main__':
    save_list = ['just', 'for', 'test']
    save_dict = {'just': 0, 'for': 1, 'test': 2}
    list_save(save_list, '1.txt')
    load_list = list_load('1.txt')
    print(load_list)
    dict_save(save_dict, '2.txt')
    load_dict = dict_load('2.txt')
    print(load_dict)
