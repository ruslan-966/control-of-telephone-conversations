import pandas as pd
import pymorphy2
import re
from program_parametr import *

pd.options.mode.chained_assignment = None  # default='warn'
morph = pymorphy2.MorphAnalyzer()
hello_list = []


def sentence_into_words(string):
    if string is not None:
        return re.findall(r'[a-zа-я\d.]+', string)


def take_name(text):
    if text is not None:
        for word in text:
            for p in morph.parse(word):
                if 'Name' in p.tag and p.score >= PROB_THRESH:
                    return p.normal_form, p.score
    return None


def morphological_analysis(list_string):
    global morph
    if list_string is not None:
        words = []
        for word in list_string:
            p = morph.parse(word)[0]  # делаем разбор
            words.append(p.normal_form)
        return words


def take_words_from_text_file(path_file):
    with open(path_file, encoding='UTF-8') as f:
        all_words = f.read().splitlines()
    return all_words


def is_hello(text):
    global hello_list
    hello_list.sort(key=len, reverse=True)
    for word in hello_list:
        res = re.search(word, text)
        if res is not None:
            return res.end()


def take_greeting(text, number):
    if number > 1:
        return text[:int(number)]
    return None


def is_introducing(text):
    # обыгрываем различные варианты приветствия
    reg_exp = r"{0}\s{1}\s[а-я]+".format(INTRODUCING_FORM[0], INTRODUCING_FORM[1])
    res = re.search(reg_exp, text)
    if res is not None:
        return res[0]
    reg_exp = r"{0}\s[а-я]+\s{1}".format(INTRODUCING_FORM[0], INTRODUCING_FORM[1])
    res = re.search(reg_exp, text)
    if res is not None:
        return res[0]
    reg_exp = r"{0}\s{1}\s[а-я]+".format(INTRODUCING_FORM[1], INTRODUCING_FORM[0])
    res = re.search(reg_exp, text)
    if res is not None:
        return res[0]
    reg_exp = r"{}\s[а-я]+".format(INTRODUCING_FORM[2])
    res = re.search(reg_exp, text)
    if res is not None:
        intermediate_expression = res[0]
        lst = sentence_into_words(intermediate_expression)
        for word in lst:
            for p in morph.parse(word):
                if 'Name' in p.tag and p.score >= PROB_THRESH:
                    return intermediate_expression
    return None


def find_business_name(text):
    word = 'компания'
    ls = sentence_into_words(text)
    if word in ls:
        speech_part_list = []
        element_number = ls.index(word)     # номер слова, с которого начинается название компании
        for word in ls:
            p = morph.parse(word)[0]  # делаем разбор
            speech_part_list.append(p.tag.POS)
        last_element_number = 0
        for i in range(element_number, len(text)):
            if speech_part_list[i] not in ['ADJF', 'NOUN']:
                last_element_number = i
                break
        if last_element_number - element_number == 1 and element_number > 0:
            element_number -= 1
            test = ' '.join(ls[element_number:last_element_number])
            return test
        elif last_element_number - element_number > 1:
            test = ' '.join(ls[element_number:last_element_number])
            return test
    return None


def find_goodbye(text):
    for word in GOODBYE_LIST:
        res = re.search(word, text)
        if res is not None:
            goodbye_text = []
            ls = sentence_into_words(text)
            speech_part = []
            case_word = []
            for word_1 in ls:
                p = morph.parse(word_1)[0]  # делаем разбор
                speech_part.append(p.tag.POS)
                case_word.append(p.tag.case)
            for i in range(len(speech_part) - 1):
                cond = speech_part[i] == 'ADJF'
                cond = cond and speech_part[i + 1] == 'ADJF'
                cond = cond and case_word[i] == 'gent'
                cond = cond and case_word[i + 1] == 'gent'
                if cond:
                    goodbye_text.append(' '.join([ls[i], ls[i + 1]]))
                cond = speech_part[i] == 'PREP'
                cond = cond and speech_part[i + 1] == 'NOUN'
                cond = cond and case_word[i + 1] == 'gent'
                if cond:
                    goodbye_text.append(' '.join([ls[i], ls[i + 1]]))
            return goodbye_text
    return None


def result_cell_filling(row, result):
    if row.introduce_manager is not None:
        id_conversation = row.dlg_id
        result.at[id_conversation, 'manager_introduced_himself'] = row.introduce_manager


def do_well_definer(row):
    filter_1 = type(row['manager_said_hello']) is str
    filter_2 = type(row['manager_said_goodbye']) is list
    if filter_1 and filter_2:
        return 'Ok!'
    return 'Terrible result!'


def conversations_control():
    global hello_list
    # Открываем файл содержащий разговоры
    dt = pd.read_csv(INPUT_DATA_FILE, delimiter=',', encoding='utf-8')
    # из записи разговоров выделяем, только те строки, которые сказал менеджер:
    manager_said = dt[dt.role == 'manager']
    # в колонке text переводим буквы в нижний регистр.
    manager_said.loc[:, 'text'] = manager_said.loc[:, 'text'].map(lambda x: x.lower())
    # Определяем, как менеджер поздоровался, для этого:
    # скачиваем контрольный список слов приветствия:
    hello_list = take_words_from_text_file(GREETING_FILE)
    # находим строки, где есть приветствие и запоминаем номер последнего символа приветствия:
    manager_said['greeting_length'] = manager_said['text'].apply(is_hello)
    # копируем текст приветствия:
    manager_said['greeting_text'] = manager_said['text'].astype(object).\
        combine(manager_said['greeting_length'], func=take_greeting)
    # ищем строки, где менеджер представил себя:
    manager_said['introduce_manager'] = manager_said['text'].apply(is_introducing)
    # определяем имя менеджера, для этого:
    # нормализуем колонку text:
    # Создаем дополнительную колонку и заносим в нее список слов:
    manager_said['words'] = manager_said['introduce_manager'].apply(sentence_into_words)
    # Переводим слова в нормальную форму:
    manager_said['words'] = manager_said['words'].apply(morphological_analysis)
    # выделяем собственные имена
    manager_said['name'] = manager_said['words'].apply(take_name)
    # выделяем упоминание компании
    manager_said['company'] = manager_said['text'].apply(find_business_name)
    # выделяем, где менеджер прощался
    manager_said['goodbye'] = manager_said['text'].apply(find_goodbye)
    print(manager_said[manager_said['greeting_text'].notna()])
    # создаем датафрейм для записи результатов:
    column_names = ['conversation_id', 'manager_said_hello', 'manager_introduced_himself', 'managers_name',
                    'company_name', 'manager_said_goodbye', 'did_he_do_well']
    result = pd.DataFrame(columns=column_names)
    # начинаем заполнять result, заносим id разговоров:
    result['conversation_id'] = dt['dlg_id'].unique()
    # Просматриваем строки датафрейма manager_said, группируем и заносим информацию в result
    for row in manager_said.itertuples(index=False):
        if row.greeting_text is not None:
            id_conversation = row.dlg_id
            result.at[id_conversation, 'manager_said_hello'] = row.greeting_text
        if row.introduce_manager is not None:
            id_conversation = row.dlg_id
            result.at[id_conversation, 'manager_introduced_himself'] = row.introduce_manager
        if row.name is not None:
            id_conversation = row.dlg_id
            result.at[id_conversation, 'managers_name'] = list(row.name)[0]
        if row.company is not None:
            id_conversation = row.dlg_id
            result.at[id_conversation, 'company_name'] = row.company
        if row.goodbye is not None:
            id_conversation = row.dlg_id
            result.at[id_conversation, 'manager_said_goodbye'] = [''.join(map(str, l_)) for l_ in row.goodbye]
    # проверяем главное требование к менеджеру:
    result['did_he_do_well'] = result.apply(lambda x: do_well_definer(x), axis=1)
    with pd.ExcelWriter('output.xlsx') as writer:
        result.to_excel(writer)
    pd.set_option('display.max_columns', None)
    print(result)
    return


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    conversations_control()
