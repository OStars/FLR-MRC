import json
import random


def strQ2B(input):
    """全角->半角
    Args:
        input (str): 输入字符串
    Returns:
        str: 输出字符串
    """
    output = input.replace("“","\"").replace("”","\"")

    return output



label_name=set()

data_list=['CCKS_train.json','CCKS_dev.json']
for data_i in data_list:
    res = []
    with open(data_i,encoding='utf8') as f:
        lines=json.load(f)
        for line in lines:
            entity=[]
            if 'entities' not in line:
                res.append({"text": ' '.join(list(strQ2B(line['text']).lower())), "entities": entity})
                continue
            for i in line['entities']:
                label_name.add(i['type'])  # ' '.join(list(line['originalText'][i['start_pos']:i['end_pos']])) #line['originalText'][i['start_pos']*2:i['end_pos']*2+1]
                tmp = {'label': i['type'],
                       'text': ' '.join(strQ2B(i['entity']).lower()),
                       'start_offset': i['start_idx'] * 2,
                       'end_offset': i['end_idx'] * 2 + 1}
                entity.append(tmp)

            res.append({"text": ' '.join(list(strQ2B(line['text']).lower())), "entities": entity})


    with open(data_i.split('_')[1],'w',encoding='utf8') as outf:
        for idx,i in enumerate(res):
            outf.write(json.dumps(i,ensure_ascii=False)+'\n')

print(label_name)

