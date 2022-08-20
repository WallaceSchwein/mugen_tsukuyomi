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
term s of the license agreement you entered into with Willi Kristen's software solutions.
"""

import datetime
import json
import regex
import requests
import urllib
import xml.etree.ElementTree as ET

from bs4 import BeautifulSoup as BS
from datetime import datetime as dt

from tsukuyomi.utils.util_tools import CONFIG_PATH_SCRIPT, load_config

config = load_config(CONFIG_PATH_SCRIPT)


class DataGenerator():

    __name__ = "TsukuyomiDataGenerator"
    __version__ = "1.0.0"

    URL_BASE: str = "https://www.bundestag.de"
    URL_START: str = f"{URL_BASE}/services/opendata"
    URL_SUFFIX: str = "?limit=10&noFilterSet=true&offset="

    FORMAT = '%d.%m.%Y'

    PATTERN_DATE: regex.Pattern = regex.compile('(0[1-9]|[1-2][0-9]|3[0-1])\.(0[1-9]|1[0-2])\.(20(1[7-9]|2[0-2]))')
    PATTERN_NAME: regex.Pattern = regex.compile('(Dr. )?(\p{Lu}\p{Ll}+ \p{Lu}\p{Ll}+|'\
                                                '\p{Lu}\p{Ll}+ \p{Lu}\p{Ll}+-\p{Lu}\p{Ll}+|'\
                                                '\p{Lu}\p{Ll}+ \p{Lu}. \p{Lu}\p{Ll}+|'\
                                                '\p{Lu}\p{Ll}+ \p{Ll}+ \p{Ll}+ \p{Lu}\p{Ll}+|'\
                                                '\p{Lu}\p{Ll}+-\p{Lu}\p{Ll}+ \p{Lu}\p{Ll}+|'\
                                                '\p{Lu}\p{Ll}+ \p{Lu}\p{Ll}+ \p{Lu}\p{Ll}+|'\
                                                '\p{Lu}\p{Ll}+ \p{Ll}+ \p{Lu}\p{Ll}+|'\
                                                '\p{Lu}\p{Ll}+ \p{Ll}+ \p{Lu}\p{Ll}+ \p{Lu}\p{Ll}+|'\
                                                '\p{Lu}\p{Ll}+-\p{Lu}\p{Ll}+ \p{Lu}\p{Ll}+-\p{Lu}\p{Ll}+|'\
                                                '\p{Lu}\p{Ll}+ \p{Lu}\p{Ll}+ \p{Ll}+ \p{Lu}\p{Ll}+|'\
                                                '\p{Lu}\p{Ll}+-\p{Lu}\p{Ll}+ \p{Ll}+ \p{Ll}+ \p{Lu}\p{Ll}+|'\
                                                '\p{Lu}\p{Ll}+ \p{Lu}\p{Ll}+ \p{Lu}\p{Ll}+-\p{Lu}\p{Ll}+)')

    first_period: int = config['TIME_FRAME']['FIRST_PERIOD']
    last_period: int = config['TIME_FRAME']['LAST_PERIOD']
    oldest: datetime.date = dt.strptime("23.10.2017", FORMAT).date() if config['TIME_FRAME']['OLDEST'] == "" \
                                                                     else dt.strptime(config['TIME_FRAME']['OLDEST'], FORMAT).date()
    latest: datetime.date = datetime.date.today() if config['TIME_FRAME']['LATEST'] == "" \
                                                  else dt.strptime(config['TIME_FRAME']['LATEST'], FORMAT).date()

    no_party_fix: dict = json.load(open("scripts/setup_files/no_party_fix.json"))
    # TODO automate the set up of no_party_fix.json with an extra crawler.

    def fetch_data(self): # NOSONAR
        """
        Fetches the stenographic protocols from sessions of the german parliament 
        for extracting the single speeches later.

        Yields:
            ElementTree: XML-document parsed to an ElementTree.
        """

        r = requests.get(self.URL_START)
        k = BS(r.text, "html.parser")

        for i, item in enumerate(k.select("section.bt-module-row.bt-module-row-dokumente")):
            if i >= self.first_period and i <= self.last_period:
                section_url = item.get("data-dataloader-url")

                for offset in range(0, 310, 10):
                    url = f"{self.URL_BASE}{section_url}{self.URL_SUFFIX}{offset}"
                    r2 = requests.get(url)
                    k2 = BS(r2.text, "html.parser")
                    for item in k2.select(".bt-link-dokument"):
                        data_url = item.get("href")
                        print(f"\n\t- Requested XML-File: {self.URL_BASE}{data_url}")
                        r3 = urllib.request.urlopen(f"{self.URL_BASE}{data_url}")
                        t = r3.read().decode("utf-8")
                        xml_elem = ET.fromstring(t)

                        try:
                            tree = ET.ElementTree(xml_elem)
                        except Exception:
                            print(f"\t[ERROR] while reading file:\n{self.URL_BASE}{data_url}")
                            continue

                        yield tree

    def apply_time_frame(self, tree: ET.ElementTree):
        """
        Filters the fetched XML-documents for the date of the protocolled parliament's session.
        Ensures that only speeches from sessions within the set time frame are considered for the model data.

        Args:
            tree (ET.ElementTree): XML-document parsed to an ElementTree.

        Returns:
            int: resturns a code, wether the document is filtered out by it's date or is relevant.
        """

        root = tree.getroot()
        date_str = root.find("./vorspann/kopfdaten/veranstaltungsdaten/datum").attrib['date']
        date = dt.strptime(date_str, self.FORMAT).date()

        if date >= self.oldest and date <= self.latest:
            return 0
        elif date >= self.oldest and date >= self.latest:
            print(f"\t\t[INFO] Out of targeted time frame - Continue!")
            return 1
        elif date <= self.oldest and date <= self.latest:
            print(f"\t\t[INFO] Finished targeted time frame - End of fetching!\n")
            return 2

    def extract_speeches(self, tree: ET.ElementTree): # NOSONAR
        """
        Extracts every speech and it's meta data from a parliament's protocoll as XML
        and yields text and meta data in a dict.

        Args:
            tree (ET.ElementTree): XML-document parsed to an ElementTree.

        Yields:
            dict: The extracted speech, including meta data, such as the speaker, his party, etc.
        """

        root = tree.getroot()

        date = root.find("./vorspann/kopfdaten/veranstaltungsdaten/datum").attrib['date']
        if not regex.fullmatch(self.PATTERN_DATE, date):
            return
        meeting_no = root.find("./vorspann/kopfdaten/plenarprotokoll-nummer/sitzungsnr").text
        period = root.find("./vorspann/kopfdaten/plenarprotokoll-nummer/wahlperiode").text

        speeches = root.findall("./sitzungsverlauf/tagesordnungspunkt/rede")
        for speech_raw in speeches:

            speaker_fir = speech_raw.find("./p[@klasse='redner']/redner/name/vorname")
            speaker_fam = speech_raw.find("./p[@klasse='redner']/redner/name/nachname")
            speaker = ""
            if speaker_fir is None:
                continue
            elif speaker_fam is None:
                continue
            else: 
                speaker: str = speaker_fir.text + " " + speaker_fam.text
                if len(speaker) <= 7 or "präsident" in speaker.lower() or not regex.fullmatch(self.PATTERN_NAME, speaker):
                    continue

            party_raw = speech_raw.find("./p[@klasse='redner']/redner/name/fraktion")
            if party_raw is not None:
                party = party_raw.text
                if party in ["BÜNDNIS\xa090/DIE GRÜNEN", "Bündnis 90/Die Grünen"]:
                    party = "BÜNDNIS 90/DIE GRÜNEN"
                if party not in ['AfD', 'BÜNDNIS 90/DIE GRÜNEN', 'CDU/CSU', 'DIE LINKE', 'FDP', 'SPD']:
                    continue
            else:
                if speaker in self.no_party_fix:
                    party = self.no_party_fix[speaker]
                else:
                    with open("scripts/setup_files/no_party_data.txt", "a+") as f:
                        no_party_curr = [rep.lower().strip() for rep in f.readlines()]
                        if speaker not in no_party_curr:
                            f.write(speaker + "\n")
                    continue

            speech: str = ""
            speaker_id = speech_raw.find("./p[@klasse='redner']/redner").attrib['id']
            relevant = True
            for i, child in enumerate(speech_raw):
                if child.tag == 'p': 
                    # TODO define filter conditions to filter answering if speaker permits interrogation
                    try:
                        if child.attrib['klasse'] != 'redner' and relevant:
                            try: 
                                speech += " " + child.text
                            except TypeError:
                                print(f"\t[WARNING - no Text]@:\n\t\t{date}: BT{period} - Meeting {meeting_no}, speaker: {speaker} - row: {i}\n")
                                continue
                        elif child.attrib['klasse'] == 'redner' and child[0].attrib['id'] == speaker_id:
                            relevant = True
                        elif child.attrib['klasse'] == 'redner' and child[0].attrib['id'] != speaker_id:
                            relevant = False
                    except KeyError:
                        print(f"\t[WARNING - no Key 'Klasse']@:\n\t\t{date}: BT{period} - Meeting {meeting_no}, speaker: {speaker} - row: {i}\n")
                        continue
                elif child.tag == 'name':
                    relevant = False

            if len(speech) < 2000 or len(speech) > 10000:
                continue

            data_dict = {"date": date,
                        "meeting_no": meeting_no,
                        "period": period,
                        "speaker": speaker,
                        "party": party,
                        "text": speech}

            yield data_dict


    def create_model_data(self):
        """
        Creates data for model training from the stenographic protocols of the german parliament, 
        offered by it's open data service @https://bundestag.de/service/opendata.

        Returns:
            list: List of data objects, containing the extracted parliament's speeches.
        """

        speeches = []

        for elem in self.fetch_data():
            match self.apply_time_frame(elem):
                case 0:
                    for speech in self.extract_speeches(elem):
                        speeches.append(speech)
                case 1:
                    continue
                case 2:
                    break
                case _:
                    print("[ERROR] An unknown error occured during fetching! Please check, if time the frame is set correctly..")

        return speeches