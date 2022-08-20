"""
tsukuyomi microservice
The python implementation of helpful and somehow generic functions

Created: April 2022
@author: Willi Kristen

@license: Willi Kristen:
Copyright (c) 2022 Willi Kristen, Germany
https://de.linkedin.com/in/willi-kristen-406887218

All rights reserved, also regarding any disposal, exploitation, reproduction, editing, distribution.
This software is the confidential and proprietary information of Willi Kristen.
You shall not disclose such confidential information and shall use it only in accordance with the
terms of the license agreement you entered into with Willi Kristen's software solutions.
"""

import os
import regex
import wget
import xml.etree.ElementTree as ET

from nltk.corpus import stopwords
from typing import List
from zipfile import ZipFile

from tsukuyomi.utils.util_tools import CONFIG_PATH_SCRIPT, load_config

config = load_config(CONFIG_PATH_SCRIPT)


class FilterGenerator():

    __name__ = "TsukuyomiFilterGenerator"
    __version__ = "1.0.0"

    URL_REP = "https://www.bundestag.de/resource/blob/472878/7615ae8b1b4881b303004610edcd49a0/MdB-Stammdaten-data.zip"

    INCORRECT_ENTRIES = ["been", "bes", "bälde", "ei,", "einbaün", "en", "for", "hen", "kaeumlich", "ncht",
                         "sa", "solc", "o g", "o ä", "u a", "u g", "u ä", "z b", "d h", "hä?", "und?"]

    def normal_stopwords(self):
        """
        Creates stop word list, of german stop words, to filter out in preprocessing.

        Returns:
            list: German stop word list.
        """

        print(f"\n\tBeginning to create normal stop word list...\n")

        with open("./scripts/setup_files/stop_words_raw_material.txt", "r") as f:
            other_stop_words = [word.lower().strip() for word in f.readlines()]

        nltk_stop_words = stopwords.words('german')

        stop_words: List[str] = sorted(list(set(other_stop_words + nltk_stop_words)))

        for entry in self.INCORRECT_ENTRIES:
            stop_words.remove(entry)

        with open("tsukuyomi/business_layer/preprocessing/preprocessing_files/stop_words_de.txt", "w") as sw:
            for word in stop_words:
                sw.write(word + "\n")

        return stop_words


    def special_stopwords(self):
        """
        Creates a list of german stop words, in a parliamentary context, 
        such as the names of representants or political parties e.g.

        Returns:
            tuple[list, list]: Tuple, containing two lists: 
                First: Names of german representatives.
                Second: Additional stop words in context of the parliament's speeches.
        """
        print(f"\n\tBeginning to create special parliament's stop word list...\n")

        with open("tsukuyomi/business_layer/preprocessing/preprocessing_files/representatives.txt", "r") as r:
            representatives = [word.lower().strip() for word in r.readlines()]
        with open("tsukuyomi/business_layer/preprocessing/preprocessing_files/additional_terms.txt", "r") as a:
            additionals = [word.lower().strip() for word in a.readlines()]

        rep_zip = wget.download(self.URL_REP)
        with ZipFile(rep_zip, 'r') as zip:
            zip.printdir()
            zip.extractall()
        os.remove(rep_zip)
        os.remove("MDB_STAMMDATEN.DTD")

        tree = ET.parse("MDB_STAMMDATEN.XML")
        root = tree.getroot()

        mdbs = root.findall("./MDB")
        for mdb in mdbs:
            mdb_fir = mdb.find("./NAMEN/NAME/VORNAME")
            mdb_fam = mdb.find("./NAMEN/NAME/NACHNAME")

            mdb_periods = mdb.findall("./WAHLPERIODEN/WAHLPERIODE")

            for period in mdb_periods:
                if period.find("./WP").text in ["19", "20"]:
                    if mdb_fam.text.lower() != "merkel" and mdb_fam.text.lower() != "scholz":
                        for name in mdb_fam.text.split():
                            if name not in representatives:
                                representatives.append(name.lower().strip())
                        for name in mdb_fir.text.split():
                            if name not in additionals:
                                additionals.append(name.lower().strip())
        os.remove("MDB_STAMMDATEN.XML")

        representatives = set(representatives)
        additionals = set(additionals)

        with open("tsukuyomi/business_layer/preprocessing/preprocessing_files/representatives.txt", "w") as r:
            for rep in representatives:
                r.write(rep + "\n")
        with open("tsukuyomi/business_layer/preprocessing/preprocessing_files/additional_terms.txt", "w") as a:
            for add in additionals:
                a.write(add + "\n")

        return (representatives, additionals)

    def create_stopwords(self):
        """
        Creates all stop words lists used in tsukuyomi preprocessing.

        Returns:
            tuple[list, list, list]: Tuple, containing three lists:
                First: Casual German stop words
                Second: Names of german representatives.
                Third: Additional stop words in context of the parliament's speeches.
        """

        normals = self.normal_stopwords()
        reps, adds = self.special_stopwords()

        return (normals, reps, adds)

    def load_stopwords(self):
        """
        Loads all stop words lists used in tsukuyomi preprocessing.

        Returns:
            tuple[list, list, list]: Tuple, containing three lists:
                First: Casual German stop words
                Second: Names of german representatives.
                Third: Additional stop words in context of the parliament's speeches.
        """

        print(f"\n\tLoading the stop word lists...\n")

        with open("tsukuyomi/business_layer/preprocessing/preprocessing_files/stop_words_de.txt", "r") as sw:
            normals = [word.lower().strip() for word in sw.readlines()]

        with open("tsukuyomi/business_layer/preprocessing/preprocessing_files/representatives.txt", "r") as r:
            representatives = [word.lower().strip() for word in r.readlines()]
        with open("tsukuyomi/business_layer/preprocessing/preprocessing_files/additional_terms.txt", "r") as a:
            additionals = [word.lower().strip() for word in a.readlines()]

        return (normals, representatives, additionals)

    def check_fomals(self, data: List[dict]):
        """
        Lists the most used terms in the beginning and the end of the parliament's speeches for manual check 
        and consider some of them as additional stop words in parliamentary context.

        Args:
            data (List[dict]): List of data objects, containing the extracted speeches.
        """

        print(f"\n\tBeginning to check the words of formal sentences at the beginning and the end of the fetched speeches...\n")

        with open("tsukuyomi/business_layer/preprocessing/preprocessing_files/stop_words_de.txt", "r") as sw:
            stop_words = [word.lower().strip() for word in sw.readlines()]
        with open("tsukuyomi/business_layer/preprocessing/preprocessing_files/additional_terms.txt", "r") as a:
            additionals = [word.lower().strip() for word in a.readlines()]
        formals = {}

        for elem in data:
            txt = regex.sub(regex.compile('\s+'), " ", elem['text'])
            txt = regex.split(regex.compile("[.!?:]"), txt)

            start_speech = txt[:2]
            end_speech = txt[-2:]

            for phrase in start_speech:
                for token in regex.sub(regex.compile('[,;]'), "", phrase).split():
                    if token not in stop_words and token not in additionals:
                        if token in formals:
                            formals[token] += 1
                        else:
                            formals[token] = 1
            for phrase in end_speech:
                for token in regex.sub(regex.compile('[,;]'), "", phrase).split():
                    if token not in stop_words and token not in additionals:
                        if token in formals:
                            formals[token] += 1
                        else:
                            formals[token] = 1

        formals = sorted(formals.items(), key=lambda x: x[1], reverse=True)
        formals = formals[:35]

        with open("scripts/setup_files/formals.txt", "a+") as f:
            current = [word.lower().strip() for word in f.readlines()]

            for formal in set(formals):
                if formal not in current:
                    f.write(f"{formal[0].lower().strip()}, {formal[1]}\n")
