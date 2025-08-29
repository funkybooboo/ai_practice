from typing import List

def flatten(n_d_list: List) -> List[float]:
    one_d_list: List[float] = []
    for item in n_d_list:
        if isinstance(item, list):
            one_d_list.extend(flatten(item))
        else:
            one_d_list.append(item)
    return one_d_list
