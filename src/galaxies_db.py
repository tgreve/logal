"""
This module reads in IR luminosities, CO line fluxes (Jup=1 to Jup=14), and
a number of the main far-IR fine-structure lines for several samples
of local galaxies (normal, LIRGs, ULIRGs), cross correlates the different
samples, and stores the data in a database, which can be activated and
accessed by other scripts.

Extended sources are not included in the database.

All luminosities are converted to the same cosmology:
(Ho, Omega_L, Omega_m) = (70, 0.72, 0.28)




Correct K16: L_40_120 ---> L_8_1000

Extract sources that are common to both K16 and L17

subset of sources where you have L_40_120.K16 and L_8_1000.L17

Plot L_8_1000.L17/L_40_120.K16 vs. L_40_120.K16
Mean o median of C = (L_8_1000.L17/L_40_120.K16) of if there is a trend with

Then take all of your K16 L_40_120.K16 x C ---> L_8_1000.K16





Samples:
--------
	The raw data files were downloaded from the online data servers given below
        and then stored in logal/raw-data/

	Armus+09:
	    Armus-et-al-2009-table-1.txt
	    (Ho, Omega_L, Omega_m) = (70, 0.72, 0.28)

	Greve+14:
	    Greve+14-LIR.txt
	    (Ho, Omega_L, Omega_m) = (67, 0.685, 0.315)

	Rosenberg+15:
	    Rosenberg-et-al-2015.txt
            IR[8-1000] luminosities are from Armus+09
	    (Ho, Omega_L, Omega_m) = (70, 0.72, 0.28)

        Israel+15:
            https://www.aanda.org/articles/aa/full_html/2015/06/aa25175-14/aa25175-14.html#tabs
            Israel-et-al-2015-Table-1.txt
            Israel-et-al-2015-Table-2.txt
            Israel-et-al-2015-Table-3.txt
            Israel-et-al-2015-Table-4.txt
            Israel-et-al-2015-Table-5.txt
	    (Ho, Omega_L, Omega_m) = (73, 0.73, 0.27)

	Kamenetzky+16:
	    http://vizier.cfa.harvard.edu/viz-bin/VizieR?-source=J/ApJ/829/93
	    Kamenetzky-et-al-2015-Table-1.txt
	    Kamenetzky-et-al-2015-Table-2.tsv
	    Kamenetzky-et-al-2015-Table-3.tsv
	    (Ho, Omega_L, Omega_m) = (70, 0.72, 0.28)  <--- Assumed!

	Lu+17:
	    (Ho, Omega_L, Omega_m) = (70, 0.7, 0.3)  <--- Assumed! Adopted from Jiao+17


	Jiao+17:
	    (Ho, Omega_L, Omega_m) = (70, 0.7, 0.3)
            Detected lines have positive fluxes and errors. Upper limits have
            negative fluxes, with the abs() value being the 3sigma upper limit.
            7 sources are in the extended list and will be removed.


Comments on extended sources and multiple pointings:
----------------------------------------------------
	Rosenberg+15:
                For all extended sources (Arp299, ESO173-G015, MCG+12–02–001,
                Mrk331, NGC1068 NGC1365, NGC2146, NGC3256, NGC5135, and
                NGC7771), an aperture correction is necessary to compensate for
                the wavelength dependent beam size (Makiwa et al. 2013). We
                defined a source as extended using LABOCA or SCUBA 350 or 450 μm
                (respectively) maps with 8′′ resolution. We convolved the 8′′
                resolution maps with the SPIRE FTS resolution, and if the galaxy
                was more extended than the smallest SPIRE beam size, we defined
                it as extended. In order to correct for the extended nature of
                these sources, we employ HIPE’s semiExtendedCorrector tool
                (SECT). This tool “derives” an intrinsic source size by
                iterating over different source sizes until it finds one that
                provides a good match in the overlap range of the two observing
                bands near 1000 GHz, and is further discussed in Wu et al.
                (2013). We set the Gaussian reference beam to 42′′, the largest
                SPIRE beamsize. The beamsize corrected flux values for the 10
                extended sources are listed in Table 2, along with the compact
                sources

                There are three targets in the sample that have multiple
                pointings; Arp 299, NGC 1365, and NGC 2146. In the case of Arp
                299, we use only the pointing for Arp 299 A. For NGC 1365 we
                take the average of the northeast and southwest pointings. This
                is done since the northeast and southwest pointings have
                approximately a 50% overlap in field of view at the center of
                the galaxy. This overlap region is the center of the PACS
                observations, so for comparison, it is best to average the
                northeast and southwest pointings. For NGC 2146, we use only the
                nuclear pointing.


Run Mongod database:
		>mongod --dbpath=/Users/tgreve/Dropbox/NEW-SETUP/C.Code/local/python/logal/db
                >lsof -i | grep 27017
                >db.eval("db.shutdownServer()")

                If previous instance of Mongo db is hanging:
                >sudo lsof -iTCP -sTCP:LISTEN -n -P
                >sudo kill ...


Todo:
    29.03.2020: Need to incorporate Jiao+17 data.
Status:
    29.03.2020: pkl-files have ValUnDef where there are no entries. But in the
                db, there are no ValUnDef. There are just no entries.



t.greve@ucl.ac.uk, 2017
"""


# --- Modules ---
from importlib import reload
import numpy as np
import math
import sys
import time
import pandas as pd

from astroquery.ned import Ned
from astroquery.ned.core import RemoteServiceError
from astropy.cosmology import FlatLambdaCDM, z_at_value
from astropy import units as u
from pymongo import MongoClient

import trgpy.emg
reload(trgpy.emg)
from trgpy.emg import line_flux_conversion
from trgpy.dictionary_transitions import freq
from trgpy.config import cosmo_params_standard_1
cosmo_params = cosmo_params_standard_1
# ---------------


# --- Global variables ---
# Path to local data
data_path = '/Users/tgreve/Dropbox/Work/local/python/logal/raw-data/'

# Raw data table names
table1_name_J17 = "Jiao-et-al-2017.txt"

table1_name_L17 = "Lu-et-al-2017-table-1.txt"
table4_name_L17 = "Lu-et-al-2017-table-4.txt"

table1_name_K16 = "Kamenetzky-et-al-2015-Table-1.bsv"
table2_name_K16 = "Kamenetzky-et-al-2015-Table-2.bsv"
table3_name_K16 = "Kamenetzky-et-al-2015-Table-3.bsv"

table1_name_R15 = "Rosenberg-et-al-2015-table-1.txt"
table2_name_R15 = "Rosenberg-et-al-2015-table-2.txt"
table3_name_R15 = "Rosenberg-et-al-2015-table-3.txt"

table1_name_I15 = "Israel-et-al-2015-Table-1.txt"
table2_name_I15 = "Israel-et-al-2015-Table-2.txt"
table3_name_I15 = "Israel-et-al-2015-Table-3.txt"
table4_name_I15 = "Israel-et-al-2015-Table-4.txt"
table5_name_I15 = "Israel-et-al-2015-Table-5.txt"

table_name_G14 = "Greve-et-al-2014-table-1.txt"
table1_name_A09 = "Armus-et-al-2009-table-1.txt"

# Extended sources
#extended_sources = ['M82','Arp299', 'ESO173-G015', 'MCG+12-02-001', 'MGC+12-02-001', 'Mrk331', 'NGC1068',
#                    'NGC1365', 'NGC2146','NGC3256', 'NGC5135', 'NGC7771','MilkyWay','SgrA*',
#                   'NGC3690', 'UGC06471', 'UGC06742','NGC4038','NGC4038overlap']
#
extended_sources = ['Arp299',                                  # R15, multiple pointings
                    'NGC2146',                                 # R15, multiple pointings, L17: NGC2146NW, Nuc, SE
                    'NGC1365',                                 # R15, multiple pointings
                    'NGC3690',                                 # L17
                    'UGC06471',                                # L17
                    'Arp299B',
                    'IC694',
                    'UGC06742',                                # L17
                    'NGC3690A',                                # L17
                    'Arp299A'
                    'NGC4038',
                    'NGC4038overlap',
                    'NGC5010',                                 # L17
                    'IRAS05223+1908',                          # L17
                    'MilkyWay',
                    'SgrA*',
                    'NGC3256']                                 # R15, extended source
                    #'NGC7771',                                 # R15, extended source
# Magic values
ValUnDef = -999
ValUpLim = 99
ValLoLim = -99

# Cosmology
cosmo  = {'omega_M_0': 0.28, 'omega_lambda_0': 0.72, 'omega_k_0': 0.0,'h': 0.70}
# ------------------------

# --- Connect to database db.local_galaxies ---
client = MongoClient()
db = client.master_database
collection_local_galaxies = db.local_galaxies
# ---------------------------------------------


def build():
    """ Initializes db.local_galaxies, creates pkl-files and master_list, and
        finally builds db.local_galaxies.
    """

    # Initialize
    db_initialize(drop=True)

    # Make pkl-files
    make_pickle_A09()
    make_pickle_G14()
    make_pickle_R15()
    make_pickle_I15()
    make_pickle_K16()
    make_pickle_L17()
    make_pickle_J17()

    # Make master list
    make_master_list(verbose=True)

    # Commit to db
    commit_to_db_master()
    commit_to_db_A09()
    commit_to_db_I15()
    commit_to_db_G14()
    commit_to_db_R15()
    commit_to_db_K16()
    commit_to_db_L17()
    commit_to_db_J17()


def db_initialize(drop=False):
    """ Initializes database.
    """

    # --- Initialize database db.local_galaxies ---
    client = MongoClient()
    db = client.master_database
    if drop:
        db.drop_collection('local_galaxies')
    collection_local_galaxies = db.local_galaxies
    # ---------------------------------------------


def remove_extended_sources(DataFrame):
    """Removes extended sources from dataframe
    """

    for ID in extended_sources:
        # Check whether ID is in DataFrame.ID
        indices = [i for i,x in enumerate(DataFrame.ID) if ID in x]
        if len(indices) > 0:
            DataFrame = DataFrame.drop(DataFrame.index[[indices]])

        # Check whether ID_ALT exist
        if 'ID_ALT' in DataFrame.columns:
            # Check whether ID is in DataFrame.ID_ALT
            indices = [i for i,x in enumerate(DataFrame.ID_ALT) if ID in x]
            if len(indices) > 0:
                DataFrame = DataFrame.drop(DataFrame.index[[indices]])

    DataFrame = DataFrame.reset_index(level=0, drop=True)

    return DataFrame


def map_id_raw_to_id(table, catalogue=False, drop=True):
    """ Maps (ID_RAW, ID_ALT_RAW) to ID.
    """
    # --- General trimming ---
    table.ID_RAW = [x.replace('{','') for x in table.ID_RAW]
    table.ID_RAW = [x.replace('}','') for x in table.ID_RAW]
    table.ID_RAW = [x.replace('tt','') for x in table.ID_RAW]
    table.ID_RAW = [x.replace(r""'\\','') for x in table.ID_RAW]
    table.ID_RAW = [x.replace(' ','') for x in table.ID_RAW]
    table.ID_RAW = [x.replace('\t','') for x in table.ID_RAW]
    table.ID_RAW = [x.replace('*','') for x in table.ID_RAW]

    if 'ID_ALT_RAW' in table.columns:
        table.ID_ALT_RAW = [x.replace('{','') for x in table.ID_ALT_RAW]
        table.ID_ALT_RAW = [x.replace('}','') for x in table.ID_ALT_RAW]
        table.ID_ALT_RAW = [x.replace('tt','') for x in table.ID_ALT_RAW]
        table.ID_ALT_RAW = [x.replace(r""'\\','') for x in table.ID_ALT_RAW]
        table.ID_ALT_RAW = [x.replace(' ','') for x in table.ID_ALT_RAW]
        table.ID_ALT_RAW = [x.replace('\t','') for x in table.ID_ALT_RAW]
    # ------------------------

    # --- Create a ID and ID_ALT column in dataframe ---
    table['ID'] = table.apply(lambda row: row['ID_RAW'], axis=1)
    if 'ID_ALT_RAW' in table.columns:
        table['ID_ALT'] = table.apply(lambda row: row['ID_ALT_RAW'], axis=1)
    if catalogue == 'I15':
        # Create new ID_ALT_RAW and ID_ALT column in dataframe
        table['ID_ALT_RAW'] = table.apply(lambda row: row['ID_RAW'], axis=1)
        table['ID_ALT'] = table.apply(lambda row: row['ID_ALT_RAW'], axis=1)
    if catalogue == 'J17':
        # Create new ID_ALT_RAW and ID_ALT column in dataframe
        table['ID_ALT_RAW'] = table.apply(lambda row: row['ID_RAW'], axis=1)
        table['ID_ALT'] = table.apply(lambda row: row['ID_ALT_RAW'], axis=1)
    # --------------------------------------------------

    # --- Format A09 ---
    if catalogue == 'A09':
        # Check ID_RAW. If F is missing (i.e., only phone number), then add F
        i=0
        for ID in table.ID_RAW:
            if not('F' in ID):
                table.ID_RAW.iloc[i] = "F"+ID
            i = i+1

        # If ALT_ID_RAW is nan, replace ALT_ID with ID (with IRAS added to it)
        i=0
        for ID in table.ID_ALT_RAW:
            if ID == 'nan':
                table.ID_ALT_RAW.iloc[i] = ''
                new_ID = table.ID.iloc[i]
                if not('F' in new_ID):
                    new_ID = 'IRAS'+new_ID
                new_ID = new_ID.replace('F','IRAS')
                table.ID_ALT.iloc[i] = new_ID
            if 'tablenotemark' in ID:
                ID = ID.replace('tablenotemarka','')
                table.ID_ALT_RAW.iloc[i] = ID
                table.ID_ALT.iloc[i] = ID
            i = i+1

        # Add IRAS to sources with no F
        i=0
        for ID in table.ID:
            if not('F' in ID):
                table.ID.iloc[i] = 'IRAS' + ID
            i = i+1

        # Replace F with IRAS
        table.ID = [x.replace('F','IRAS') for x in table.ID]
    # ------------------

    # --- Format I15 ---
    if catalogue == 'I15':
        # Loop over ID_RAW to infer ID_ALT_RAW
        for i in np.arange(0,len(table.ID_RAW)):
            if '/' in table.ID_RAW[i]:
                foo = table.ID_RAW[i]
                foo = foo.split("/")
                table.loc[i,('ID')] = foo[0]
                table.loc[i,('ID_ALT')] = foo[1]
            elif '(' in table.ID_RAW[i]:
                foo = table.ID_RAW[i]
                foo = foo.split("(")
                table.loc[i,('ID')] = foo[0]
                table.ID_ALT_RAW[i] = foo[1]
                table.ID_ALT_RAW = [x.replace(')','') for x in table.ID_ALT_RAW]
            else:
                table.loc[i,('ID_ALT_RAW')] = ''
    # ------------------

    # --- General (ID_RAW, ID_ALT_RAW) --> (ID, ID_ALT) formatting ---
    # ID_RAW --> ID
    # A09
    indices = [i for i,x in enumerate(table.ID) if 'IRAS05189-2524' in x]
    if len(indices) > 0:
        table.loc[indices,('ID')] = 'IRASF05189-2524'
    indices = [i for i,x in enumerate(table.ID) if 'IRAS18293-3413' in x]
    if len(indices) > 0:
        table.loc[indices,('ID')] = 'IRASF18293-3413'
    indices = [i for i,x in enumerate(table.ID) if '13120-5453' in x]
    if len(indices) > 0:
        table.loc[indices,('ID')] = 'IRAS13120-5453'
    indices = [i for i,x in enumerate(table.ID) if '16516-0948' in x]
    if len(indices) > 0:
        table.loc[indices,('ID')] = 'IRASF16516-0948'
    indices = [i for i,x in enumerate(table.ID) if '16399-0937' in x]
    if len(indices) > 0:
        table.loc[indices,('ID')] = 'IRASF16399-0937'
    indices = [i for i,x in enumerate(table.ID) if '17138-1017' in x]
    if len(indices) > 0:
        table.loc[indices,('ID')] = 'IRASF17138-1017'
    indices = [i for i,x in enumerate(table.ID) if '10565+2448' in x]
    if len(indices) > 0:
        table.loc[indices,('ID')] = 'IRASF10565+2448'
    # G14
    indices = [i for i,x in enumerate(table.ID) if 'NGC34' in x]
    if len(indices) > 0:
        table.loc[indices,('ID')] = 'NGC0034'
    indices = [i for i,x in enumerate(table.ID) if '17208-0014' in x]
    if len(indices) > 0:
        table.loc[indices,('ID')] = 'IRASF17207-0014'
    indices = [i for i,x in enumerate(table.ID) if '10565+2448' in x]
    if len(indices) > 0:
        table.loc[indices,('ID')] = 'IRASF10565+2448'
    indices = [i for i,x in enumerate(table.ID) if 'IRAS02512+1446' in x]
    if len(indices) > 0:
        table.loc[indices,('ID')] = 'MCG+02-08-029'
    indices = [i for i,x in enumerate(table.ID) if 'IRAS09320+6134' in x]
    if len(indices) > 0:
        table.loc[indices,('ID')] = 'UGC05101'
    indices = [i for i,x in enumerate(table.ID) if 'IRAS12243-0036' in x]
    if len(indices) > 0:
        table.loc[indices,('ID')] = 'NGC4418'
    indices = [i for i,x in enumerate(table.ID) if 'IRAS13001-2339' in x]
    if len(indices) > 0:
        table.loc[indices,('ID')] = 'ESO507-G070'
    indices = [i for i,x in enumerate(table.ID) if 'IRAS13470+3530' in x]
    if len(indices) > 0:
        table.loc[indices,('ID')] = 'UGC08739'
    indices = [i for i,x in enumerate(table.ID) if 'IRAS15163+4255' in x]
    if len(indices) > 0:
        table.loc[indices,('ID')] = 'VV705'
    indices = [i for i,x in enumerate(table.ID) if 'IRAS15327+2340' in x]
    if len(indices) > 0:
        table.loc[indices,('ID')] = 'Arp220'
    indices = [i for i,x in enumerate(table.ID) if 'IRAS15437+0234' in x]
    if len(indices) > 0:
        table.loc[indices,('ID')] = 'NGC5990'
    indices = [i for i,x in enumerate(table.ID) if 'IRAS16284+0411' in x]
    if len(indices) > 0:
        table.loc[indices,('ID')] = 'CGCG052-037'
    indices = [i for i,x in enumerate(table.ID) if 'IRAS13188+0036' in x]
    if len(indices) > 0:
        table.loc[indices,('ID')] = 'NGC5104'
    # R15
    indices = [i for i,x in enumerate(table.ID) if 'Arp299' in x]
    if len(indices) > 0:
        table.loc[indices,('ID')] = 'Arp299-A'
    indices = [i for i,x in enumerate(table.ID) if 'Zw049.057' in x]
    if len(indices) > 0:
        table.loc[indices,('ID')] = 'CGCG049-057'
    indices = [i for i,x in enumerate(table.ID) if 'IC4687' in x]
    if len(indices) > 0:
        table.loc[indices,('ID')] = 'IC4687'
    # L17
    indices = [i for i,x in enumerate(table.ID) if '09913' in x]
    if len(indices) > 0:
        table.loc[indices,('ID')] = 'Arp220'
    indices = [i for i,x in enumerate(table.ID) if '08058' in x]
    if len(indices) > 0:
        table.loc[indices,('ID')] = 'Mrk231'
    indices = [i for i,x in enumerate(table.ID) if '03608' in x]
    if len(indices) > 0:
        table.loc[indices,('ID')] = 'UGC3608'
    indices = [i for i,x in enumerate(table.ID) if '08387' in x]
    if len(indices) > 0:
        table.loc[indices,('ID')] = 'Arp193'
    indices = [i for i,x in enumerate(table.ID) if '08696' in x]
    if len(indices) > 0:
        table.loc[indices,('ID')] = 'Mrk273'
    indices = [i for i,x in enumerate(table.ID) if '16381190' in x]
    if len(indices) > 0:
        table.loc[indices,('ID')] = 'ESO069-IG006'
    # I15
    indices = [i for i,x in enumerate(table.ID) if 'NGC1275(PerA)' in x]
    if len(indices) > 0:
        table.loc[indices,('ID')] = 'NGC1275'
    indices = [i for i,x in enumerate(table.ID) if 'MGC+12-02-001' in x]
    if len(indices) > 0:
        table.loc[indices,('ID')] = 'MCG+12-02-001'
    # K16
    indices = [i for i,x in enumerate(table.ID) if 'NGC3410a' in x]
    if len(indices) > 0:
        table.loc[indices,('ID')] = 'NGC3410'
    indices = [i for i,x in enumerate(table.ID) if 'NGC0232a' in x]
    if len(indices) > 0:
        table.loc[indices,('ID')] = 'NGC0232'
    indices = [i for i,x in enumerate(table.ID) if 'NGC3110a' in x]
    if len(indices) > 0:
        table.loc[indices,('ID')] = 'NGC3110'
    indices = [i for i,x in enumerate(table.ID) if 'NGC2388a' in x]
    if len(indices) > 0:
        table.loc[indices,('ID')] = 'NGC2388'
    indices = [i for i,x in enumerate(table.ID) if 'NGC2342b' in x]
    if len(indices) > 0:
        table.loc[indices,('ID')] = 'NGC2341'
    indices = [i for i,x in enumerate(table.ID) if 'NGC2342a' in x]
    if len(indices) > 0:
        table.loc[indices,('ID')] = 'NGC2342'
    indices = [i for i,x in enumerate(table.ID) if 'IC4518ABa' in x]
    if len(indices) > 0:
        table.loc[indices,('ID')] = 'IC4518A'
    indices = [i for i,x in enumerate(table.ID) if 'M101_02' in x]
    if len(indices) > 0:
        table.loc[indices,('ID')] = 'M101'
    indices = [i for i,x in enumerate(table.ID) if 'NGC2976_00' in x]
    if len(indices) > 0:
        table.loc[indices,('ID')] = 'NGC2976'
    indices = [i for i,x in enumerate(table.ID) if 'MCG+04-48-002a' in x]
    if len(indices) > 0:
        table.loc[indices,('ID')] = 'MCG+04-48-002'
    indices = [i for i,x in enumerate(table.ID) if 'NGC5734a' in x]
    if len(indices) > 0:
        table.loc[indices,('ID')] = 'NGC5734'
    indices = [i for i,x in enumerate(table.ID) if 'NGC7679a' in x]
    if len(indices) > 0:
        table.loc[indices,('ID')] = 'NGC7679'
    indices = [i for i,x in enumerate(table.ID) if 'IC10-B11-1' in x]
    if len(indices) > 0:
        table.loc[indices,('ID')] = 'IC10'
    indices = [i for i,x in enumerate(table.ID) if 'NGC0877a' in x]
    if len(indices) > 0:
        table.loc[indices,('ID')] = 'NGC0877'
    indices = [i for i,x in enumerate(table.ID) if 'NGC891-1' in x]
    if len(indices) > 0:
        table.loc[indices,('ID')] = 'NGC891'
    indices = [i for i,x in enumerate(table.ID) if 'NGC205-copeak' in x]
    if len(indices) > 0:
        table.loc[indices,('ID')] = 'NGC205'

    # ID_ALT_RAW --> ID_ALT
    if 'ID_ALT_RAW' in table.columns:
        indices = [i for i,x in enumerate(table.ID_ALT) if 'MRK0331' in x]
        if len(indices) > 0:
            table.loc[indices,('ID_ALT')] = 'Mrk331'
        indices = [i for i,x in enumerate(table.ID_ALT) if '09913' in x]
        if len(indices) > 0:
            table.loc[indices,('ID_ALT')] = 'Arp220'
        indices = [i for i,x in enumerate(table.ID_ALT) if '08058' in x]
        if len(indices) > 0:
            table.loc[indices,('ID_ALT')] = 'Mrk231'
        indices = [i for i,x in enumerate(table.ID_ALT) if '08387' in x]
        if len(indices) > 0:
            table.loc[indices,('ID_ALT')] = 'Arp193'
        indices = [i for i,x in enumerate(table.ID_ALT) if '08696' in x]
        if len(indices) > 0:
            table.loc[indices,('ID_ALT')] = 'Mrk273'
        indices = [i for i,x in enumerate(table.ID_ALT) if 'Arp256' in x]
        if len(indices) > 0:
            table.loc[indices,('ID_ALT')] = 'MCG-02-01-051'
        indices = [i for i,x in enumerate(table.ID_ALT) if '02369' in x]
        if len(indices) > 0:
            table.loc[indices,('ID_ALT')] = 'MCG+02-08-029'
    # ----------------------------------------------------------------

    # --- Remove duplicates ---
    if drop:
        table = table.drop_duplicates(subset='ID')
        table = table.reset_index(level=0, drop=True)
        if 'ID_ALT' in table.columns:
            table = table.drop_duplicates(subset='ID_ALT')
            table = table.reset_index(level=0, drop=True)
    # -------------------------

    return table


def make_pickle_A09():
    """ Creates A09 pkl file.
    """

    # --- Read in source ID_RAW and ID_ALT_RAW from A09 ---
    table = pd.read_csv(data_path+table1_name_A09, skiprows=33, delimiter='&',
            skipinitialspace=True, names=['ID_RAW', 'ID_ALT_RAW', 'LIR_8_1000'],
            usecols=[0,1,6])


    # Make sure ID_ALT_RAW is str type (if not, can be issues with NaN)
    table.ID_ALT_RAW = table.ID_ALT_RAW.astype(str)
    # -----------------------------------------------------

    # --- Map (ID_RAW, ID_ALT_RAW) to (ID) ---
    table = map_id_raw_to_id(table, catalogue='A09')
    # ----------------------------------------

    # --- Remove extended sources ---
    table = remove_extended_sources(table)
    # -------------------------------

    # --- Query NED for redshifts ---
    z = []
    for i in np.arange(0,len(table.ID_RAW)):
        result_table = Ned.query_object(table.ID_RAW[i])
        z_ned=result_table['Redshift']
        if z_ned > -1:
            z.append(z_ned[0])
        else:
            ID = table.ID[i]
            result_table = Ned.query_object(ID)
            z_ned=result_table['Redshift']
            z.append(z_ned[0])

        _progressBar("Armus+09 sample: ", i, len(table.ID))
    # -------------------------------

    # --- Fix LIR and convert to standard cosmology ---
    for i in np.arange(0,len(table.ID_RAW)):
        x = table.LIR_8_1000[i]
        x = x.replace('tt','')
        x = x.replace(r""'\\','')
        x = x.replace('{','')
        x = x.replace('}','')
        x = x.replace('phm','')
        x = x.replace(':','')
        table.LIR_8_1000.iloc[i] = 10.**float(x)
    # -------------------------------------------------

    # --- Pickle data ---
    df = pd.DataFrame({'ID_RAW': table.ID_RAW, 'ID_ALT_RAW': table.ID_ALT_RAW,
        'ID': table.ID, 'ID_ALT': table.ID_ALT, 'z': z, 'LIR_8_1000':
        table.LIR_8_1000})
    df.to_pickle(data_path+'Armus-et-al-2009.pkl')
    # -------------------


def make_pickle_G14():
    """ Creates G14 pkl file.
    """

    # --- Read in source ID_RAW and ID_ALT_RAW from A09 ---
    table = pd.read_csv(data_path+table_name_G14, skiprows=1,
            delim_whitespace=True,skipinitialspace=True,
            names=['ID_RAW', 'z','LIR_50_300', 'LIR_8_1000'])
    # -----------------------------------------------------

    # --- Map (ID_RAW, ID_ALT_RAW) to (ID) ---
    table = map_id_raw_to_id(table)
    # ----------------------------------------

    # --- Remove extended sources ---
    table = remove_extended_sources(table)
    # -------------------------------

    # --- Fix LIR and convert into standard cosmology ---
    cosmo = FlatLambdaCDM(H0=100.*cosmo_params['h'],
            Om0=cosmo_params['omega_M_0'], Tcmb0=cosmo_params['Tcmb0'])
    d_L = cosmo.luminosity_distance(table.z)
    d_L = d_L.value
    cosmo_params_G14  = {'omega_M_0': 0.315, 'omega_lambda_0': 0.685,
                  'omega_k_0': 0.0, 'h': 0.67, 'Tcmb0': 2.725}
    cosmo_G14 = FlatLambdaCDM(H0=100.*cosmo_params_G14['h'],
            Om0=cosmo_params_G14['omega_M_0'], Tcmb0=cosmo_params_G14['Tcmb0'])
    d_L_G14 = cosmo_G14.luminosity_distance(table.z)
    d_L_G14 = d_L_G14.value

    correction_factor = (d_L/d_L_G14)**2.

    table.loc[:,('LIR_8_1000')] = (10.**table.LIR_8_1000)*correction_factor
    table.loc[:,('LIR_50_300')] = (10.**table.LIR_50_300)*correction_factor
    # ---------------------------------------------------

    # --- Pickle data ---
    df = pd.DataFrame({'ID_RAW': table.ID_RAW, 'ID': table.ID, 'z': table.z,
                       'LIR_50_300': table.LIR_50_300,
                       'LIR_8_1000': table.LIR_8_1000})
    df.to_pickle(data_path+'Greve-et-al-2014.pkl')
    # -------------------

    time.sleep(2)
    _progressBar("Greve+14 sample: ", 1, 1)


def make_pickle_R15():
    """ Creates R15 pkl file.
    """

    # --- Read in Table 1 from Rosenberg+15 ---
    table1 = pd.read_csv(data_path+table1_name_R15, skiprows=4, delimiter='&',
                         skipinitialspace=True, names=['ID_RAW', 'logIR_8_1000',
                                                       'FIR', 'z', 'DL',
                                                       'FWHM_CO10',
                                                       'AGN_SB_type'],
                         usecols=[0,1,2,3,4,5,6])

    # Map (ID_RAW, ID_ALT_RAW) to (ID)
    table1 = map_id_raw_to_id(table1)

    # Remove extended sources
    table1 = remove_extended_sources(table1)

    # Fix IR luminosities and rename column
    for i in np.arange(0,len(table1.ID_RAW)):
        table1.loc[i,('logIR_8_1000')] = 10.**table1.logIR_8_1000.loc[i]

    table1.rename(columns = {'logIR_8_1000':'LIR_8_1000'}, inplace=True)
    # --------------------------------------------


    # --- Read in Table 2 from Rosenberg+15 ---
    table2 = pd.read_csv(data_path+table2_name_R15, skiprows=20, delimiter='&',
            skipinitialspace=True, names=['ID_RAW', 'SdV_CO43', 'SdV_CO54',
                'SdV_CO65', 'SdV_CO76', 'SdV_CO87', 'SdV_CO98', 'SdV_CO109',
                'SdV_CO1110', 'SdV_CO1211', 'SdV_CO1312', 'SdV_CI609',
                'SdV_CI370', 'SdV_OI63', 'SdV_OI145', 'SdV_CII158'])

    # Map (ID_RAW, ID_ALT_RAW) to (ID)
    table2 = map_id_raw_to_id(table2)

    # Remove extended sources
    table2 = remove_extended_sources(table2)

    # Convert entries to float
    entries = ['43','54','65','76','87','98','109','1110','1211','1312']
    entries = ['CO' + s for s in entries]
    entries = ['SdV_' + s for s in entries]
    entries = entries + ['SdV_CI609','SdV_CI370','SdV_OI63', 'SdV_OI145',
                         'SdV_CII158']
    for entry in entries:
        table2[entry] = [str(x).replace('\t','') if '\t' in str(x) else str(x) for x in table2[entry]]
    # --------------------------------------------


    # --- Read in Table 3 from Rosenberg+15 ---
    table3 = pd.read_csv(data_path+table3_name_R15, skiprows=12, delimiter='&',
                         skipinitialspace=True, names=['ID_RAW', 'SdV_CO10',
                                                       'Beam_CO10',
                                                       'Reference_CO10',
                                                       'SdV_CO21', 'Beam_CO21',
                                                       'Reference_CO21',
                                                       'SdV_CO32', 'Beam_CO32',
                                                       'Reference_CO32'])

    # Map (ID_RAW, ID_ALT_RAW) to (ID)
    table3 = map_id_raw_to_id(table3)

    # Remove extended sources
    table3 = remove_extended_sources(table3)

    # Convert entries to float
    entries = ['10','21','32']
    entries = ['CO' + s for s in entries]
    entries = ['SdV_' + s for s in entries]
    for entry in entries:
        table3[entry] = [str(x).replace('\t','') if '\t' in str(x) else str(x) for x in table3[entry]]


    # --- Pickle data ---
    df = pd.DataFrame({'ID_RAW': table1.ID_RAW, 'ID': table1.ID, 'z': table1.z,
        'LIR_8_1000': table1.LIR_8_1000, 'FIR': table1.FIR, 'AGN_SB_type':
        table1.AGN_SB_type, 'SdV_CO10': table3.SdV_CO10, 'SdV_CO21':
        table3.SdV_CO21, 'SdV_CO32': table3.SdV_CO32, 'SdV_CO43':
        table2.SdV_CO43, 'SdV_CO54': table2.SdV_CO54, 'SdV_CO65':
        table2.SdV_CO65, 'SdV_CO76': table2.SdV_CO76, 'SdV_CO87':
        table2.SdV_CO87, 'SdV_CO98': table2.SdV_CO98, 'SdV_CO109':
        table2.SdV_CO109, 'SdV_CO1110': table2.SdV_CO1110, 'SdV_CO1211':
        table2.SdV_CO1211, 'SdV_CO1312': table2.SdV_CO1312, 'SdV_CI609':
        table2.SdV_CI609, 'SdV_CI370': table2.SdV_CI370, 'SdV_OI63':
        table2.SdV_OI63, 'SdV_OI145': table2.SdV_OI145, 'SdV_CII158':
        table2.SdV_CII158})
    df.to_pickle(data_path+'Rosenberg-et-al-2015.pkl')
    # -------------------

    time.sleep(2)
    _progressBar("Rosenberg+15 sample: ", 1, 1)


def make_pickle_I15():
    """ Creates I15 pkl file.
    """

    _progressBar("Israel+15 sample: ", 0, 5)

    # --- Read in Table 1 from Israel+15 ---
    table1 = pd.read_csv(data_path+table1_name_I15, skiprows=3, delimiter='&',
                         skipinitialspace=True, names=['ID_RAW', 'VLSR',
                                                       'DL', 'LIR_8_1000'],
                         usecols=[0,3,4,6])

    # Map (ID_RAW, ID_ALT_RAW) to (ID)
    table1 = map_id_raw_to_id(table1, catalogue='I15')

    # Remove extended sources
    table1 = remove_extended_sources(table1)
    # --------------------------------------------

    # --- Get redshift for given DL ---
    cosmo_params_I15  = {'omega_M_0': 0.27, 'omega_lambda_0': 0.73,
            'omega_k_0': 0.0, 'h': 0.73, 'Tcmb0': 2.73}
    cosmo_I15 = FlatLambdaCDM(H0=100.*cosmo_params_I15['h'],
            Om0=cosmo_params_I15['omega_M_0'], Tcmb0=cosmo_params_I15['Tcmb0'])
    z = np.ones(len(table1.DL))
    for i in np.arange(0,len(table1.DL)):
        z[i] = z_at_value(cosmo_I15.luminosity_distance, table1.DL.values[i]*u.Mpc, zmax=1.0)
    # ---------------------------------

    # --- Fix LIR and convert into standard cosmology ---
    cosmo = FlatLambdaCDM(H0=100.*cosmo_params['h'],
            Om0=cosmo_params['omega_M_0'], Tcmb0=cosmo_params['Tcmb0'])
    d_L = cosmo.luminosity_distance(z)

    d_L_I15 =cosmo_I15.luminosity_distance(z)

    correction_factor = (d_L/d_L_I15)**2.

    table1.loc[:,('LIR_8_1000')] = (10.**table1.LIR_8_1000)*correction_factor
    # ---------------------------------------------------
    _progressBar("Israel+15 sample: ", 1, 5)


    # --- Read in Table 2 from Israel+15 ---
    table2 = pd.read_csv(data_path+table2_name_I15, skiprows=18, delimiter='&',
                         skipinitialspace=True, names=['ID_RAW', 'SdV_CO43',
                                                       'SdV_CO76', 'SdV_CI609',
                                                       'SdV_CI370', 'SdV_CO21',
                                                       'SdV_13CO21'],
                         usecols=[0,1,2,3,4,5,6])

    # Map (ID_RAW, ID_ALT_RAW) to (ID)
    table2 = map_id_raw_to_id(table2, catalogue='I15')

    # Remove extended sources
    table2 = remove_extended_sources(table2)

    # Trim \t etc
    table2.SdV_CO43 = [x.replace('\t','') for x in table2.SdV_CO43]
    table2.SdV_CO43 = [ValUnDef if '−' in str(x) else x for x in table2.SdV_CO43]
    table2.SdV_CO21 = [x.replace('\t','') for x in table2.SdV_CO21]
    table2.SdV_CO21 = [ValUnDef if '−' in str(x) else x for x in table2.SdV_CO21]
    table2.SdV_13CO21 = [x.replace('\t','') for x in table2.SdV_13CO21]
    table2.SdV_13CO21 = [ValUnDef if '−' in str(x) else x for x in table2.SdV_13CO21]

    table2.SdV_CO43 = table2.SdV_CO43.astype(float)
    table2.SdV_CO76 = table2.SdV_CO76.astype(float)
    table2.SdV_CO21 = table2.SdV_CO21.astype(float)
    table2.SdV_13CO21 = table2.SdV_13CO21.astype(float)
    table2.SdV_CI609 = table2.SdV_CI609.astype(float)
    table2.SdV_CI370 = table2.SdV_CI370.astype(float)

    # Convert to [W m^-2]
    for i in np.arange(0,len(table2.ID_RAW)):
        if table2.loc[i,('SdV_CO43')] != ValUnDef:
            table2.loc[i,('SdV_CO43')] = table2.SdV_CO43.loc[i]*1.E-17
        if table2.loc[i,('SdV_CO76')] != ValUnDef:
            table2.loc[i,('SdV_CO76')] = table2.SdV_CO76.loc[i]*1.E-17
        if table2.loc[i,('SdV_CI609')] != ValUnDef:
            table2.loc[i,('SdV_CI609')] = table2.SdV_CI609.loc[i]*1.E-17
        if table2.loc[i,('SdV_CI370')] != ValUnDef:
            table2.loc[i,('SdV_CI370')] = table2.SdV_CI370.loc[i]*1.E-17
        if table2.loc[i,('SdV_CO21')] != ValUnDef:
            table2.loc[i,('SdV_CO21')] = table2.SdV_CO21.loc[i]*1.E-19
        if table2.loc[i,('SdV_13CO21')] != ValUnDef:
            table2.loc[i,('SdV_13CO21')] = table2.SdV_13CO21.loc[i]*1.E-19


    # Add 15% errors
    table2['eSdV_CO43'] = table2.apply(lambda row: 0.15*row['SdV_CO43'], axis=1)
    table2.eSdV_CO43 = [ValUnDef if x < 0 else x for x in table2.eSdV_CO43]
    table2['eSdV_CO76'] = table2.apply(lambda row: 0.15*row['SdV_CO76'], axis=1)
    table2.eSdV_CO76 = [ValUnDef if x < 0 else x for x in table2.eSdV_CO76]
    table2['eSdV_CO21'] = table2.apply(lambda row: 0.20*row['SdV_CO21'], axis=1)
    table2.eSdV_CO21 = [ValUnDef if x < 0 else x for x in table2.eSdV_CO21]
    table2['eSdV_13CO21'] = table2.apply(lambda row: 0.20*row['SdV_13CO21'], axis=1)
    table2.eSdV_13CO21 = [ValUnDef if x < 0 else x for x in table2.eSdV_13CO21]
    table2['eSdV_CI609'] = table2.apply(lambda row: 0.15*row['SdV_CI609'], axis=1)
    table2.eSdV_CI609 = [ValUnDef if x < 0 else x for x in table2.eSdV_CI609]
    table2['eSdV_CI370'] = table2.apply(lambda row: 0.15*row['SdV_CI370'], axis=1)
    table2.eSdV_CI370 = [ValUnDef if x < 0 else x for x in table2.eSdV_CI370]
    # --------------------------------------------
    _progressBar("Israel+15 sample: ", 2, 5)

    # --- Read in Table 3 from Israel+15 ---
    table3 = pd.read_csv(data_path+table3_name_I15, skiprows=18, delimiter='&',
                         skipinitialspace=True, names=['ID_RAW', 'SdV_CO43',
                                                       'SdV_CO76', 'SdV_CI609',
                                                       'SdV_CI370', 'SdV_CO21',
                                                       'SdV_13CO21'],
                         usecols=[0,1,2,3,4,5,6])

    # Map (ID_RAW, ID_ALT_RAW) to (ID)
    table3 = map_id_raw_to_id(table3, catalogue='I15')

    # Remove extended sources
    table3 = remove_extended_sources(table3)

    # Trim etc
    table3.SdV_CO21 = [ValUnDef if '−' in str(x) else x for x in table3.SdV_CO21]
    table3.SdV_13CO21 = [ValUnDef if '−' in str(x) else x for x in table3.SdV_13CO21]

    table3.SdV_CO43 = table3.SdV_CO43.astype(float)
    table3.SdV_CO76 = table3.SdV_CO76.astype(float)
    table3.SdV_CO21 = table3.SdV_CO21.astype(float)
    table3.SdV_13CO21 = table3.SdV_13CO21.astype(float)
    table3.SdV_CI609 = table3.SdV_CI609.astype(float)
    table3.SdV_CI370 = table3.SdV_CI370.astype(float)

    for i in np.arange(0,len(table3.ID_RAW)):
        if table3.loc[i,('SdV_CO43')] != ValUnDef:
            table3.loc[i,('SdV_CO43')] = table3.SdV_CO43.loc[i]*1.E-17
        if table3.loc[i,('SdV_CO76')] != ValUnDef:
            table3.loc[i,('SdV_CO76')] = table3.SdV_CO76.loc[i]*1.E-17
        if table3.loc[i,('SdV_CI609')] != ValUnDef:
            table3.loc[i,('SdV_CI609')] = table3.SdV_CI609.loc[i]*1.E-17
        if table3.loc[i,('SdV_CI370')] != ValUnDef:
            table3.loc[i,('SdV_CI370')] = table3.SdV_CI370.loc[i]*1.E-17
        if table3.loc[i,('SdV_CO21')] != ValUnDef:
            table3.loc[i,('SdV_CO21')] = table3.SdV_CO21.loc[i]*1.E-19
        if table3.loc[i,('SdV_13CO21')] != ValUnDef:
            table3.loc[i,('SdV_13CO21')] = table3.SdV_13CO21.loc[i]*1.E-19

    # Add 15% errors
    table3['eSdV_CO43'] = table3.apply(lambda row: 0.15*row['SdV_CO43'], axis=1)
    table3.eSdV_CO43 = [ValUnDef if x < 0 else x for x in table3.eSdV_CO43]
    table3['eSdV_CO76'] = table3.apply(lambda row: 0.15*row['SdV_CO76'], axis=1)
    table3.eSdV_CO76 = [ValUnDef if x < 0 else x for x in table3.eSdV_CO76]
    table3['eSdV_CO21'] = table3.apply(lambda row: 0.20*row['SdV_CO21'], axis=1)
    table3.eSdV_CO21 = [ValUnDef if x < 0 else x for x in table3.eSdV_CO21]
    table3['eSdV_13CO21'] = table3.apply(lambda row: 0.20*row['SdV_13CO21'], axis=1)
    table3.eSdV_13CO21 = [ValUnDef if x < 0 else x for x in table3.eSdV_13CO21]
    table3['eSdV_CI609'] = table3.apply(lambda row: 0.15*row['SdV_CI609'], axis=1)
    table3.eSdV_CI609 = [ValUnDef if x < 0 else x for x in table3.eSdV_CI609]
    table3['eSdV_CI370'] = table3.apply(lambda row: 0.15*row['SdV_CI370'], axis=1)
    table3.eSdV_CI370 = [ValUnDef if x < 0 else x for x in table3.eSdV_CI370]
    # --------------------------------------------
    _progressBar("Israel+15 sample: ", 3, 5)

    # --- Read in Table 4 from Israel+15 ---
    table4 = pd.read_csv(data_path+table4_name_I15, skiprows=19, delimiter='&',
                         skipinitialspace=True, names=['ID_RAW', 'SdV_CO43',
                                                       'SdV_CO76', 'SdV_CI609',
                                                       'SdV_CI370', 'SdV_CO21',
                                                       'SdV_13CO21'],
                         usecols=[0,1,2,3,4,5,6])

    # Map (ID_RAW, ID_ALT_RAW) to (ID)
    table4 = map_id_raw_to_id(table4, catalogue='I15')

    # Remove extended sources
    table4 = remove_extended_sources(table4)

    # Trim etc
    table4.SdV_CO43 = [x.replace('\t','') for x in table4.SdV_CO43]
    table4.SdV_CO76 = [x.replace('\t','') for x in table4.SdV_CO76]
    table4.SdV_CO21 = [x.replace('\t','') for x in table4.SdV_CO21]
    table4.SdV_13CO21 = [x.replace('\t','') for x in table4.SdV_13CO21]
    table4.SdV_CI609 = [x.replace('\t','') for x in table4.SdV_CI609]
    table4.SdV_CO43 = [ValUnDef if '−' in str(x) else x for x in table4.SdV_CO43]
    table4.SdV_CO76 = [ValUnDef if '−' in str(x) else x for x in table4.SdV_CO76]
    table4.SdV_CO21 = [ValUnDef if '−' in str(x) else x for x in table4.SdV_CO21]
    table4.SdV_13CO21 = [ValUnDef if '−' in str(x) else x for x in table4.SdV_13CO21]
    table4.SdV_13CO21 = [ValUnDef if '—' in str(x) else x for x in table4.SdV_13CO21]
    table4.SdV_CI609 = [ValUnDef if '−' in str(x) else x for x in table4.SdV_CI609]

    table4.SdV_CO43 = table4.SdV_CO43.astype(float)
    table4.SdV_CO76 = table4.SdV_CO76.astype(float)
    table4.SdV_CO21 = table4.SdV_CO21.astype(float)
    table4.SdV_13CO21 = table4.SdV_13CO21.astype(float)
    table4.SdV_CI609 = table4.SdV_CI609.astype(float)
    table4.SdV_CI370 = table4.SdV_CI370.astype(float)

    for i in np.arange(0,len(table4.ID)):
        if table4.loc[i,('SdV_CO43')] != ValUnDef:
            table4.loc[i,('SdV_CO43')] = table4.SdV_CO43.loc[i]*1.E-17
        if table4.loc[i,('SdV_CO76')] != ValUnDef:
            table4.loc[i,('SdV_CO76')] = table4.SdV_CO76.loc[i]*1.E-17
        if table4.loc[i,('SdV_CI609')] != ValUnDef:
            table4.loc[i,('SdV_CI609')] = table4.SdV_CI609.loc[i]*1.E-17
        if table4.loc[i,('SdV_CI370')] != ValUnDef:
            table4.loc[i,('SdV_CI370')] = table4.SdV_CI370.loc[i]*1.E-17
        if table4.loc[i,('SdV_CO21')] != ValUnDef:
            table4.loc[i,('SdV_CO21')] = table4.SdV_CO21.loc[i]*1.E-19
        if table4.loc[i,('SdV_13CO21')] != ValUnDef:
            table4.loc[i,('SdV_13CO21')] = table4.SdV_13CO21.loc[i]*1.E-19

    # Add 15% errors
    table4['eSdV_CO43'] = table4.apply(lambda row: 0.15*row['SdV_CO43'], axis=1)
    table4.eSdV_CO43 = [ValUnDef if x < 0 else x for x in table4.eSdV_CO43]
    table4['eSdV_CO76'] = table4.apply(lambda row: 0.15*row['SdV_CO76'], axis=1)
    table4.eSdV_CO76 = [ValUnDef if x < 0 else x for x in table4.eSdV_CO76]
    table4['eSdV_CO21'] = table4.apply(lambda row: 0.20*row['SdV_CO21'], axis=1)
    table4.eSdV_CO21 = [ValUnDef if x < 0 else x for x in table4.eSdV_CO21]
    table4['eSdV_13CO21'] = table4.apply(lambda row: 0.20*row['SdV_13CO21'], axis=1)
    table4.eSdV_13CO21 = [ValUnDef if x < 0 else x for x in table4.eSdV_13CO21]
    table4['eSdV_CI609'] = table4.apply(lambda row: 0.15*row['SdV_CI609'], axis=1)
    table4.eSdV_CI609 = [ValUnDef if x < 0 else x for x in table4.eSdV_CI609]
    table4['eSdV_CI370'] = table4.apply(lambda row: 0.15*row['SdV_CI370'], axis=1)
    table4.eSdV_CI370 = [ValUnDef if x < 0 else x for x in table4.eSdV_CI370]
    # --------------------------------------------
    _progressBar("Israel+15 sample: ", 4, 5)

    # --- Read in Table 5 from Israel+15 ---
    table5 = pd.read_csv(data_path+table5_name_I15, skiprows=17, delimiter='&',
                         skipinitialspace=True, names=['ID_RAW', 'SdV_CO43',
                                                       'SdV_CO76', 'SdV_CI609',
                                                       'SdV_CI370', 'SdV_CO21',
                                                       'SdV_13CO21'],
                         usecols=[0,1,2,3,4,5,6])

    # Map (ID_RAW, ID_ALT_RAW) to (ID)
    table5 = map_id_raw_to_id(table5, catalogue='I15')

    # Remove extended sources
    table5 = remove_extended_sources(table5)

    # Trim etc
    table5.SdV_CO43 = [x.replace('\t','') for x in table5.SdV_CO43]
    table5.SdV_CO76 = [x.replace('\t','') for x in table5.SdV_CO76]
    table5.SdV_13CO21 = [x.replace('\t','') for x in table5.SdV_13CO21]
    table5.SdV_CO43 = [ValUnDef if '−' in str(x) else x for x in table5.SdV_CO43]
    table5.SdV_CO76 = [ValUnDef if '−' in str(x) else x for x in table5.SdV_CO76]
    table5.SdV_13CO21 = [ValUnDef if '−' in str(x) else x for x in table5.SdV_13CO21]
    table5.SdV_CI370 = [ValUnDef if '−' in str(x) else x for x in table5.SdV_CI609]


    table5.SdV_CO43 = table5.SdV_CO43.astype(float)
    table5.SdV_CO76 = table5.SdV_CO76.astype(float)
    table5.SdV_CO21 = table5.SdV_CO21.astype(float)
    table5.SdV_13CO21 = table5.SdV_13CO21.astype(float)
    table5.SdV_CI609 = table5.SdV_CI609.astype(float)
    table5.SdV_CI370 = table5.SdV_CI370.astype(float)

    for i in np.arange(0,len(table5.ID_RAW)):
        if table5.loc[i,('SdV_CO43')] != ValUnDef:
            table5.loc[i,('SdV_CO43')] = table5.SdV_CO43.loc[i]*1.E-17
        if table5.loc[i,('SdV_CO76')] != ValUnDef:
            table5.loc[i,('SdV_CO76')] = table5.SdV_CO76.loc[i]*1.E-17
        if table5.loc[i,('SdV_CI609')] != ValUnDef:
            table5.loc[i,('SdV_CI609')] = table5.SdV_CI609.loc[i]*1.E-17
        if table5.loc[i,('SdV_CI370')] != ValUnDef:
            table5.loc[i,('SdV_CI370')] = table5.SdV_CI370.loc[i]*1.E-17
        if table5.loc[i,('SdV_CO21')] != ValUnDef:
            table5.loc[i,('SdV_CO21')] = table5.SdV_CO21.loc[i]*1.E-19
        if table5.loc[i,('SdV_13CO21')] != ValUnDef:
            table5.loc[i,('SdV_13CO21')] = table5.SdV_13CO21.loc[i]*1.E-19

    # Add 15% errors
    table5['eSdV_CO43'] = table5.apply(lambda row: 0.15*row['SdV_CO43'], axis=1)
    table5.eSdV_CO43 = [ValUnDef if x < 0 else x for x in table5.eSdV_CO43]
    table5['eSdV_CO76'] = table5.apply(lambda row: 0.15*row['SdV_CO76'], axis=1)
    table5.eSdV_CO76 = [ValUnDef if x < 0 else x for x in table5.eSdV_CO76]
    table5['eSdV_CO21'] = table5.apply(lambda row: 0.20*row['SdV_CO21'], axis=1)
    table5.eSdV_CO21 = [ValUnDef if x < 0 else x for x in table5.eSdV_CO21]
    table5['eSdV_13CO21'] = table5.apply(lambda row: 0.20*row['SdV_13CO21'], axis=1)
    table5.eSdV_13CO21 = [ValUnDef if x < 0 else x for x in table5.eSdV_13CO21]
    table5['eSdV_CI609'] = table5.apply(lambda row: 0.15*row['SdV_CI609'], axis=1)
    table5.eSdV_CI609 = [ValUnDef if x < 0 else x for x in table5.eSdV_CI609]
    table5['eSdV_CI370'] = table5.apply(lambda row: 0.15*row['SdV_CI370'], axis=1)
    table5.eSdV_CI370 = [ValUnDef if x < 0 else x for x in table5.eSdV_CI370]
    # --------------------------------------------
    _progressBar("Israel+15 sample: ", 5, 5)

    # --- Pickle Table 1 ---
    df = pd.DataFrame({'ID_RAW': table1.ID_RAW, 'ID_ALT_RAW': table1.ID_ALT_RAW,
                       'ID': table1.ID, 'ID_ALT': table1.ID_ALT, 'z': z,
                       'LIR_8_1000': table1.LIR_8_1000})
    df.to_pickle(data_path+'Israel-et-al-2015-Table-1.pkl')
    # ----------------------
    # --- Pickle Table 2 ---
    df = pd.DataFrame({'ID_RAW': table2.ID_RAW,
                       'ID_ALT_RAW': table2.ID_ALT_RAW,
                       'ID': table2.ID,
                       'ID_ALT': table2.ID_ALT,
                       'SdV_CO43': table2.SdV_CO43,
                       'SdV_CO76': table2.SdV_CO76,
                       'SdV_CI609': table2.SdV_CI609,
                       'SdV_CI370': table2.SdV_CI370,
                       'SdV_CO21': table2.SdV_CO21,
                       'SdV_13CO21': table2.SdV_13CO21,
                       'eSdV_CO43': table2.eSdV_CO43,
                       'eSdV_CO76': table2.eSdV_CO76,
                       'eSdV_CI609': table2.eSdV_CI609,
                       'eSdV_CI370': table2.eSdV_CI370,
                       'eSdV_CO21': table2.eSdV_CO21,
                       'eSdV_13CO21': table2.eSdV_13CO21})
    df.to_pickle(data_path+'Israel-et-al-2015-Table-2.pkl')
    # ----------------------
    # --- Pickle Table 3 ---
    df = pd.DataFrame({'ID_RAW': table3.ID_RAW,
                       'ID_ALT_RAW': table3.ID_ALT_RAW,
                       'ID': table3.ID,
                       'ID_ALT': table3.ID_ALT,
                       'SdV_CO43': table3.SdV_CO43,
                       'SdV_CO76': table3.SdV_CO76,
                       'SdV_CI609': table3.SdV_CI609,
                       'SdV_CI370': table3.SdV_CI370,
                       'SdV_CO21': table3.SdV_CO21,
                       'SdV_13CO21': table3.eSdV_13CO21,
                       'eSdV_CO43': table3.eSdV_CO43,
                       'eSdV_CO76': table3.eSdV_CO76,
                       'eSdV_CI609': table3.eSdV_CI609,
                       'eSdV_CI370': table3.eSdV_CI370,
                       'eSdV_CO21': table3.eSdV_CO21,
                       'eSdV_13CO21': table3.eSdV_13CO21})
    df.to_pickle(data_path+'Israel-et-al-2015-Table-3.pkl')
    # ----------------------
    # --- Pickle Table 4 ---
    df = pd.DataFrame({'ID_RAW': table4.ID_RAW,
                       'ID_ALT_RAW': table4.ID_ALT_RAW,
                       'ID': table4.ID,
                       'ID_ALT': table4.ID_ALT,
                       'SdV_CO43': table4.SdV_CO43,
                       'SdV_CO76': table4.SdV_CO76,
                       'SdV_CI609': table4.SdV_CI609,
                       'SdV_CI370': table4.SdV_CI370,
                       'SdV_CO21': table4.SdV_CO21,
                       'SdV_13CO21': table4.SdV_13CO21,
                       'eSdV_CO43': table4.eSdV_CO43,
                       'eSdV_CO76': table4.eSdV_CO76,
                       'eSdV_CI609': table4.eSdV_CI609,
                       'eSdV_CI370': table4.eSdV_CI370,
                       'eSdV_CO21': table4.eSdV_CO21,
                       'eSdV_13CO21': table4.eSdV_13CO21})
    df.to_pickle(data_path+'Israel-et-al-2015-Table-4.pkl')
    # ----------------------
    # --- Pickle Table 5 ---
    df = pd.DataFrame({'ID_RAW': table5.ID_RAW,
                       'ID_ALT_RAW': table5.ID_ALT_RAW,
                       'ID': table5.ID,
                       'ID_ALT': table5.ID_ALT,
                       'SdV_CO43': table5.SdV_CO43,
                       'SdV_CO76': table5.SdV_CO76,
                       'SdV_CI609': table5.SdV_CI609,
                       'SdV_CI370': table5.SdV_CI370,
                       'SdV_CO21': table5.SdV_CO21,
                       'SdV_13CO21': table5.SdV_13CO21,
                       'eSdV_CO43': table5.eSdV_CO43,
                       'eSdV_CO76': table5.eSdV_CO76,
                       'eSdV_CI609': table5.eSdV_CI609,
                       'eSdV_CI370': table5.eSdV_CI370,
                       'eSdV_CO21': table5.eSdV_CO21,
                       'eSdV_13CO21': table5.eSdV_13CO21})
    df.to_pickle(data_path+'Israel-et-al-2015-Table-5.pkl')
    # ----------------------



def make_pickle_K16():
    """ Creates K16 pkl file.
    """

    # === Table 1
    # --- Read in Table 1 from Kamenetzky+16 ---
    table1 = pd.read_csv(data_path+table1_name_K16,skiprows=1, sep='|',
                         skipinitialspace=True, names=['ID_RAW','LIR_40_120','DL','z'])

    # Calculate LIR_40_120
    table1.z = [0.0023 if x < 0 else x for x in table1.z]
    table1.LIR_40_120 = 10.**(table1.LIR_40_120)

    # Map (ID_RAW, ID_ALT_RAW) to (ID)
    table1 = map_id_raw_to_id(table1, drop=False)

    # Remove extended sources
    table1 = remove_extended_sources(table1)
    # ------------------------------------------


    # === Table 2
    # --- Read in raw Table 2 from Kamenetzky+16 ---
    table2 = pd.read_csv(data_path+table2_name_K16, skiprows=45, sep='|',
                             skipinitialspace=True, names=['ID_RAW','transition',
                                                           'resolved','SdV',
                                                           'SdV_1sigma_low',
                                                           'SdV_1sigma_high',
                                                           'SdV_3sigma_ul'])
    # Trim transitions
    table2.transition = table2.transition.str.strip()

    # Map (ID_RAW, ID_ALT_RAW) to (ID)
    table2 = map_id_raw_to_id(table2, drop=False)

    # Remove extended sources
    table2 = remove_extended_sources(table2)
    # ----------------------------------------------

    # --- Fix transition identification ---
    i=0
    for transition in table2.transition:
        if 'CO' in transition:
            front = '12'+transition[transition.find('CO'):transition.find('CO')+2]
            tail = transition[transition.find('CO')+2:]
            transition = front+'('+tail+')'
            table2.loc[i,'transition'] = transition
        if 'CI1' in transition:
            table2.loc[i,'transition'] = '[CI]609'
        if 'CI2' in transition:
            table2.loc[i,'transition'] = '[CI]370'
        if 'NII' in transition:
            table2.loc[i,'transition'] = '[NII]205'
        i=i+1
    # -------------------------------------

    # === Table 3
    # --- read in raw Table 3 .tsv file with line data (Jy km/s units)
    table3 = pd.read_csv(data_path+table3_name_K16, skiprows=49,sep='|',
            skipinitialspace=True,names=['ID_RAW','transition','Rflux','sigmam','sigmac',
                'x_RFlux', 'dv','Omegab','SdV','eSdV','r_RFlux'])

    # Convert to str and clean
    table3["ID_RAW"] = table3["ID_RAW"].astype(str)
    table3.ID_RAW = table3.ID_RAW.str.strip()
    table3.ID_RAW = [x.replace(' ','') for x in table3.ID_RAW]

    # Map (ID_RAW, ID_ALT_RAW) to (ID)
    table3 = map_id_raw_to_id(table3, drop=False)

    # Remove extended sources
    table3 = remove_extended_sources(table3)
    # -------------------------------------

    # --- Fix transition identification ---
    table3["transition"] = table3["transition"].astype(str)
    i=0
    for transition in table3.transition:
        if '1' in transition:
            table3.loc[i,'transition'] = "12CO(1-0)"
        if '2' in transition:
            table3.loc[i,'transition'] = "12CO(2-1)"
        if '3' in transition:
            table3.loc[i,'transition'] = "12CO(3-2)"
        if '4' in transition:
            table3.loc[i,'transition'] = "12CO(4-3)"
        if '6' in transition:
            table3.loc[i,'transition'] = "12CO(6-5)"
        if '7' in transition:
            table3.loc[i,'transition'] = "12CO(7-6)"
        i=i+1
    # -------------------------------------

    # Remove rows with SdV = NaN
    #table3 = table3[table3['SdV'].notna()]
    #table3.reset_index(drop=True)
    # -------------------------------------


    # --- Commit CO(4-3) --> CO(13-12) to db ---
    transitions = ['1-0','2-1','3-2','4-3','5-4','6-5','7-6','8-7','9-8','10-9','11-10','12-11',
                   '13-12']
    transitions = ['12CO(' + s + ')' for s in transitions]
    transitions = transitions + ['[CI]609','[CI]370','[NII]205']

    entries = ['10','21','32','43','54','65','76','87','98','109','1110','1211','1312']
    entries = ['CO' + s for s in entries]
    entries = entries + ['CI609','CI370','NII205']

    q=0
    for transition in transitions:
        if q < 3:
            table1['SdV_'+entries[q]] = table1.apply(lambda row: ValUnDef, axis=1)
            table1['eSdV_'+entries[q]] = table1.apply(lambda row: ValUnDef, axis=1)
            table1['SdV_3sigmaUL_'+entries[q]] = table1.apply(lambda row: ValUnDef, axis=1)

            # Extract entries or transition
            foo = table3[table3.transition == transition]
            foo.reset_index(drop=True)
            foo["SdV"] = foo["SdV"].astype(float)
            foo["eSdV"] = foo["eSdV"].astype(float)

            i=0
            for ID in table1.ID:
                indices_ID = [k for k,x in enumerate(foo.ID) if x == ID]
                if len(indices_ID) == 1:
                    SdV = foo.SdV.iloc[indices_ID[0]]
                    eSdV = foo.eSdV.iloc[indices_ID[0]]

                    # Only add valid values to pickle file (otherwise ValUnDef)
                    if SdV > 0:
                        table1.loc[i, ('SdV_'+entries[q])] = SdV
                    if eSdV > 0:
                        table1.loc[i, ('eSdV_'+entries[q])] = eSdV
                if len(indices_ID) > 1:
                    bar = foo.iloc[indices_ID]
                    SdV = np.mean(bar.SdV) #NaN values do not affect the mean
                    eSdV = np.std(bar.SdV) #NaN values do not affect the std
                    table1.loc[i, ('SdV_'+entries[q])] = SdV
                    table1.loc[i, ('eSdV_'+entries[q])] = eSdV
                i=i+1
        else:
            table1['SdV_'+entries[q]] = table1.apply(lambda row: ValUnDef, axis=1)
            table1['eSdV_'+entries[q]] = table1.apply(lambda row: ValUnDef, axis=1)
            table1['SdV_3sigmaUL_'+entries[q]] = table1.apply(lambda row: ValUnDef, axis=1)
            foo = table2[table2.transition == transition]
            N3 = len(foo)
            for j in range(0, N3):
                ID = foo.ID.iloc[j]
                SdV = foo.SdV.iloc[j]

                # Flux errors (e_high-e_low)/2
                SdV_1sigma_low = foo.SdV_1sigma_low.iloc[j]
                SdV_1sigma_high = foo.SdV_1sigma_high.iloc[j]
                if SdV_1sigma_high == SdV_1sigma_low:
                    eSdV = SdV
                else:
                    eSdV = (SdV_1sigma_high - SdV_1sigma_low)/2.

                # Get upper limit
                SdV_3sigma_ul = foo.SdV_3sigma_ul.iloc[j]

                indices_ID = [k for k,x in enumerate(table1.ID) if x == ID]
                if len(indices_ID) > 0:
                    if SdV > 0:
                        table1.loc[indices_ID, ('SdV_'+entries[q])] = SdV
                    if eSdV > 0:
                        table1.loc[indices_ID, ('eSdV_'+entries[q])] = eSdV
                    if SdV_3sigma_ul > 0:
                        table1.loc[indices_ID, ('SdV_3sigmaUL_'+entries[q])] = SdV_3sigma_ul
                else:
                    print("Error...")
        q=q+1


    # Replacing all occurences of NaN to -999
    table1 = table1.replace('NaN', str(ValUnDef))

    # --- Pickle Table 1 ---
    df = pd.DataFrame({'ID_RAW': table1.ID_RAW,'ID': table1.ID, 'z': table1.z,
        'LIR_40_120': table1.LIR_40_120, 'SdV_CO10': table1.SdV_CO10,
        'SdV_CO21': table1.SdV_CO32,'SdV_CO32': table1.SdV_CO32,
        'SdV_CO43': table1.SdV_CO43, 'SdV_CO54': table1.SdV_CO54,
        'SdV_CO65': table1.SdV_CO65, 'SdV_CO76': table1.SdV_CO76,
        'SdV_CO87': table1.SdV_CO87, 'SdV_CO98': table1.SdV_CO98,
        'SdV_CO109': table1.SdV_CO109, 'SdV_CO1110': table1.SdV_CO1110,
        'SdV_CO1211': table1.SdV_CO1211, 'SdV_CO1312': table1.SdV_CO1312,
        'SdV_CI609': table1.SdV_CI609, 'SdV_CI370': table1.SdV_CI370,
        'SdV_NII205': table1.SdV_NII205, 'eSdV_CO10': table1.eSdV_CO10,
        'eSdV_CO21': table1.eSdV_CO21, 'eSdV_CO32': table1.eSdV_CO32,
        'eSdV_CO43': table1.eSdV_CO43, 'eSdV_CO54': table1.eSdV_CO54,
        'eSdV_CO65': table1.eSdV_CO65, 'eSdV_CO76': table1.eSdV_CO76,
        'eSdV_CO87': table1.eSdV_CO87, 'eSdV_CO98': table1.eSdV_CO98,
        'eSdV_CO109': table1.eSdV_CO109, 'eSdV_CO1110': table1.eSdV_CO1110,
        'eSdV_CO1211': table1.eSdV_CO1211, 'eSdV_CO1312': table1.eSdV_CO1312,
        'eSdV_CI609': table1.eSdV_CI609, 'eSdV_CI370': table1.eSdV_CI370,
        'eSdV_NII205': table1.eSdV_NII205,
        'SdV_3sigmaUL_CO10': table1.SdV_3sigmaUL_CO10,
        'SdV_3sigmaUL_CO21': table1.SdV_3sigmaUL_CO21,
        'SdV_3sigmaUL_CO32': table1.SdV_3sigmaUL_CO32,
        'SdV_3sigmaUL_CO43': table1.SdV_3sigmaUL_CO43,
        'SdV_3sigmaUL_CO54': table1.SdV_3sigmaUL_CO54,
        'SdV_3sigmaUL_CO65': table1.SdV_3sigmaUL_CO65,
        'SdV_3sigmaUL_CO76': table1.SdV_3sigmaUL_CO76,
        'SdV_3sigmaUL_CO87': table1.SdV_3sigmaUL_CO87,
        'SdV_3sigmaUL_CO98': table1.SdV_3sigmaUL_CO98,
        'SdV_3sigmaUL_CO109': table1.SdV_3sigmaUL_CO109,
        'SdV_3sigmaUL_CO1110': table1.SdV_3sigmaUL_CO1110,
        'SdV_3sigmaUL_CO1211': table1.SdV_3sigmaUL_CO1211,
        'SdV_3sigmaUL_CO1312': table1.SdV_3sigmaUL_CO1312,
        'SdV_3sigmaUL_CI609': table1.SdV_3sigmaUL_CI609,
        'SdV_3sigmaUL_CI370': table1.SdV_3sigmaUL_CI370,
        'SdV_3sigmaUL_NII205': table1.SdV_3sigmaUL_NII205})
    df.to_pickle(data_path+'Kamenetzky-et-al-2016.pkl')


def make_pickle_L17():
    """ Creates L17 pkl file.
    """

    # --- Read in Table 1 from Lu+17 ---
    table1 = pd.read_csv(data_path+table1_name_L17, delimiter='&',
                         names=['ID_RAW','logLIR_8_1000','pair','C60', 'DL'],
                         usecols=[0,3,4,5,6])

    # Map (ID_RAW, ID_ALT_RAW) to (ID)
    table1 = map_id_raw_to_id(table1)

    # Remove extended sources
    table1 = remove_extended_sources(table1)

    # Query NED for redshifts
    z = []
    correction_factor = []
    time.sleep(2)
    for i in np.arange(0,len(table1.ID_RAW)):
        ID = table1.ID[i]
        result_table = Ned.query_object(ID)
        z_NED=result_table['Redshift']
        z.append(z_NED[0])
        _progressBar("Lu+17 sample: ", i, len(table1.ID_RAW))

        # Fix IR luminosities and rename column
        cosmo = FlatLambdaCDM(H0=100.*cosmo_params['h'],
                Om0=cosmo_params['omega_M_0'], Tcmb0=cosmo_params['Tcmb0'])
        d_L = cosmo.luminosity_distance(z[i])
        d_L = d_L.value

        cosmo_params_L17  = {'omega_M_0': 0.3, 'omega_lambda_0': 0.7,
                      'omega_k_0': 0.0, 'h': 0.70, 'Tcmb0': 2.725}
        cosmo_L17 = FlatLambdaCDM(H0=100.*cosmo_params_L17['h'],
                Om0=cosmo_params_L17['omega_M_0'], Tcmb0=cosmo_params_L17['Tcmb0'])
        d_L_L17 = cosmo_L17.luminosity_distance(z[i])
        d_L_L17 = d_L_L17.value

        correction_factor.append((d_L/d_L_L17)**2.)


    for i in np.arange(0,len(table1.ID_RAW)):
        try:
            table1.loc[i,('logIR_8_1000')] = (10.**float(table1.logLIR_8_1000.loc[i]))*correction_factor[i]
        except:
            table1.loc[i,('logIR_8_1000')] = ValUnDef
    table1.rename(columns = {'logIR_8_1000':'LIR_8_1000'}, inplace=True)


    # Check C60 is defined
    for i in np.arange(0,len(table1.ID_RAW)):
        try:
            foo = float(table1.C60[i]) * 10.
        except:
            table1.loc[i,('C60')] = ValUnDef


    # --- Read in Table 4 from Lu+17 ---
    table = pd.read_csv(data_path+table4_name_L17, delimiter='&', names=['ID_RAW','SdV_CO43','SdV_CO54','SdV_CO65','SdV_CO76','SdV_CO87','SdV_CO98','SdV_CO109','SdV_CO1110',
                                                                         'SdV_CO1211','SdV_CO1312','SdV_CI609','SdV_CI370',
                                                                         'SdV_NII205','eSdV_CO43','eSdV_CO54','eSdV_CO65',
                                                                         'eSdV_CO76','eSdV_CO87','eSdV_CO98','eSdV_CO109',
                                                                         'eSdV_CO1110','eSdV_CO1211','eSdV_CO1312','eSdV_CI609',
                                                                         'eSdV_CI370','eSdV_NII205','f_35','f_30','f_17'], usecols=[0,1,2,3,4,5,6,7,8,
                                                                                                               9,10,11,12,13,14,
                                                                                                               15,16,17,18,19,20,
                                                                                                               21, 22,23,24,25,26,118,119,120])

    # Map (ID_RAW, ID_ALT_RAW) to (ID)
    table = map_id_raw_to_id(table)

    # Remove extended sources
    table = remove_extended_sources(table)

    ## Remove f_35 < 0.8 sources
    #indices = [i for i,x in enumerate(table.f_30) if x < 0.8]
    #if len(indices) > 0:
    #    table = table.drop(table.index[[indices]])
    #    table1 = table1.drop(table1.index[[indices]])
    #    table = table.reset_index(level=0, drop=True)
    #    table1 = table1.reset_index(level=0, drop=True)
    #    z = [i for j,i in enumerate(z) if j not in indices]

    # Aperture correct line fluxes
    for i in np.arange(0,len(table.ID_RAW)):
        if table.SdV_CO43.loc[i] > 0:
            table.loc[i,('SdV_CO43')] = table.SdV_CO43.loc[i]/table.f_35.loc[i]
        if table.SdV_CI609.loc[i] > 0:
            table.loc[i,('SdV_CI609')] = table.SdV_CI609.loc[i]/table.f_35.loc[i]
        if table.SdV_CO54.loc[i] > 0:
            table.loc[i,('SdV_CO54')] = table.SdV_CO54.loc[i]/table.f_35.loc[i]
        if table.SdV_CO65.loc[i] > 0:
            table.loc[i,('SdV_CO65')] = table.SdV_CO65.loc[i]/table.f_30.loc[i]
        if table.SdV_CO76.loc[i] > 0:
            table.loc[i,('SdV_CO76')] = table.SdV_CO76.loc[i]/table.f_35.loc[i]
        if table.SdV_CI370.loc[i] > 0:
            table.loc[i,('SdV_CI370')] = table.SdV_CI370.loc[i]/table.f_35.loc[i]
        if table.SdV_CO87.loc[i] > 0:
            table.loc[i,('SdV_CO87')] = table.SdV_CO87.loc[i]/table.f_35.loc[i]
        if table.SdV_CO98.loc[i] > 0:
            table.loc[i,('SdV_CO98')] = table.SdV_CO98.loc[i]/table.f_17.loc[i]
        if table.SdV_CO109.loc[i] > 0:
            table.loc[i,('SdV_CO109')] = table.SdV_CO109.loc[i]/table.f_17.loc[i]
        if table.SdV_CO1110.loc[i] > 0:
            table.loc[i,('SdV_CO1110')] = table.SdV_CO1110.loc[i]/table.f_17.loc[i]
        if table.SdV_CO1211.loc[i] > 0:
            table.loc[i,('SdV_CO1211')] = table.SdV_CO1211.loc[i]/table.f_17.loc[i]
        if table.SdV_NII205.loc[i] > 0:
            table.loc[i,('SdV_NII205')] = table.SdV_NII205.loc[i]/table.f_17.loc[i]
        if table.SdV_CO1312.loc[i] > 0:
            table.loc[i,('SdV_CO1312')] = table.SdV_CO1312.loc[i]/table.f_17.loc[i]
    # -------------------------------------

    # --- Pickle data ---
    df = pd.DataFrame({'ID_RAW': table1.ID_RAW,'ID': table1.ID, 'z': z, 'LIR_8_1000': table1.LIR_8_1000,
                       'C60': table1.C60, 'SdV_CO43': table.SdV_CO43, 'SdV_CO54': table.SdV_CO54,
                       'SdV_CO65': table.SdV_CO65, 'SdV_CO76': table.SdV_CO76, 'SdV_CO87': table.SdV_CO87,
                       'SdV_CO98': table.SdV_CO98, 'SdV_CO109': table.SdV_CO109, 'SdV_CO1110': table.SdV_CO1110,
                       'SdV_CO1211': table.SdV_CO1211, 'SdV_CO1312': table.SdV_CO1312, 'SdV_CI609': table.SdV_CI609,
                       'SdV_CI370': table.SdV_CI370, 'SdV_NII205': table.SdV_NII205, 'eSdV_CO43': table.eSdV_CO43,
                       'eSdV_CO54': table.eSdV_CO54, 'eSdV_CO65': table.eSdV_CO65, 'eSdV_CO76': table.eSdV_CO76,
                       'eSdV_CO87': table.eSdV_CO87, 'eSdV_CO98': table.eSdV_CO98, 'eSdV_CO109': table.eSdV_CO109,
                       'eSdV_CO1110': table.eSdV_CO1110, 'eSdV_CO1211': table.eSdV_CO1211,
                       'eSdV_CO1312': table.eSdV_CO1312, 'eSdV_CI609': table.eSdV_CI609, 'eSdV_CI370': table.eSdV_CI370,
                       'eSdV_NII205': table.eSdV_NII205, 'f_35': table.f_35, 'f_30':table.f_30, 'f_17':table.f_17})
    df.to_pickle(data_path+'Lu-et-al-2017.pkl')
    # -------------------


#
#
def make_pickle_J17():
    """ Creates J17 pkl file.
    """

    # --- Read in Table 1 from Lu+17 ---
    table1 = pd.read_csv(data_path+table1_name_J17, delimiter='&',
                         names=['ID_RAW','ICO10','eICO10','z'],
                         usecols=[0,1,2,3], skiprows=2)

    # Clean labels
    for i in np.arange(0,len(table1)):
        foo = table1["ID_RAW"][i]
        foo = foo[0:foo.find("\t")]
        foo = foo.strip()
        table1["ID_RAW"][i] = foo

    # Map (ID_RAW, ID_ALT_RAW) to (ID)
    table1 = map_id_raw_to_id(table1, catalogue="J17")

    # Remove extended sources
    table1 = remove_extended_sources(table1)

    # --- Pickle data ---
    df = pd.DataFrame({'ID_RAW': table1.ID_RAW, 'ID': table1.ID, 'ID_ALT': table1.ID_ALT,
        'ID_ALT_RAW': table1.ID_ALT_RAW, 'z':table1.z, 'SdV_CO10': table1.ICO10, 'eSdV_CO10': table1.eICO10})
    df.to_pickle(data_path+'Jiao-et-al-2017.pkl')
    # -------------------



def make_master_list(verbose=True):

    # --- Initializations ---
    source_counter = 0
    ID_master = []
    z_master = []
    # -----------------------

    # --- Read in Armus-et-al-2009-table-1.pkl ---
    table_A09 = pd.read_pickle(data_path+'Armus-et-al-2009.pkl')

    id_A09 = table_A09.ID.values
    id_alt_A09 = table_A09.ID_ALT.values
    z_A09 = table_A09.z.values
    print("Armus+09 sample size:", len(table_A09.ID_ALT))
    print("")
    # --------------------------------------------

    # --- Read in Greve-et-al-2014-table-1.pkl ---
    table_G14 = pd.read_pickle(data_path+'Greve-et-al-2014.pkl')

    id_G14 = table_G14.ID.values
    z_G14 = table_G14.z.values
    print("Greve+14 sample size:", len(table_G14.ID))
    print("")
    # --------------------------------------------

    # --- Read in Rosenberg-et-al-2015.pkl ---
    table_R15 = pd.read_pickle(data_path+'Rosenberg-et-al-2015.pkl')

    id_R15 = table_R15.ID.values
    z_R15 = table_R15.z.values
    print("Rosenberg+15 sample size:", len(table_R15.ID))
    print("")
    # --------------------------------------------

    # --- Read in Israel-et-al-2015.pkl ---
    table_I15 = pd.read_pickle(data_path+'Israel-et-al-2015-Table-1.pkl')

    id_I15 = table_I15.ID.values
    id_alt_I15 = table_I15.ID_ALT.values
    z_I15 = table_I15.z.values
    print("Israel+15 sample size:", len(table_I15.ID))
    print("")
    # --------------------------------------------

    # --- Read in Kamenetzky-et-al-2016.pkl ---
    table_K16 = pd.read_pickle(data_path+'Kamenetzky-et-al-2016-Table-1.pkl')

    id_K16 = table_K16.ID.values
    z_K16 = table_K16.z.values
    print("Kamenetzky+16 sample size:", len(table_K16.ID))
    print("")
    # --------------------------------------------

    # --- Read in Lu-et-al-2017.pkl ---
    table_L17 = pd.read_pickle(data_path+'Lu-et-al-2017.pkl')

    id_L17 = table_L17.ID.values
    z_L17 = table_L17.z.values
    print("Lu+17 sample size:", len(table_L17.ID))
    print("")
    # --------------------------------------------

    # --- Read in Jiao-et-al-2017.pkl ---
    table_J17 = pd.read_pickle(data_path+'Jiao-et-al-2017.pkl')

    id_J17 = table_J17.ID.values
    z_J17 = table_J17.z.values
    print("Jiao+17 sample size:", len(table_J17.ID))
    print("")
    # --------------------------------------------

    # --- Output sources and sources overlaps ---
    i = 0
    if verbose:
        print("{0:4s} {1:25s} {2:25s} {3:25s} {4:25s} {5:25s} {6:25s} {7:25s} {8:25s}\n".format('No', 'K16', 'L17', 'R15', 'G14', 'A09', 'I15', 'J17', 'z'))

    # --- Loop through K16
    i_z = 0
    for name_K16 in id_K16:
        output_str = name_K16

        # Is name_K16 in L17?
        indices = [i for i,x in enumerate(id_L17) if x == name_K16]
        if len(indices) > 0:
            foo = np.array(id_L17)[indices][0]
        else:
            foo = ''
        output_str = "{0:4d} {1:25s} {2:25s}".format(source_counter, output_str, foo)


        # Is name_K16 in R15?
        indices = [i for i,x in enumerate(id_R15) if x == name_K16]
        if len(indices) > 0:
            foo = np.array(id_R15)[indices][0]
        else:
            foo = ''
        output_str = "{0:25s} {1:25s}".format(output_str, foo)


        # Is name_K16 in G14?
        indices = [i for i,x in enumerate(id_G14) if x == name_K16]
        if len(indices) > 0:
            foo = np.array(id_G14)[indices][0]
        else:
            foo = ''
        output_str = "{0:25s} {1:25s}".format(output_str, foo)


        # Is name_K16 in A09?
        indices = [i for i,x in enumerate(id_A09) if x == name_K16]
        indices_alt = [i for i,x in enumerate(id_alt_A09) if x == name_K16]
        # Yes, in .ID
        if len(indices) > 0:
            foo = np.array(id_A09)[indices][0]
            id_alt_A09 = np.array(id_alt_A09)
            id_alt_A09[indices] = name_K16
            id_alt_A09 = list(id_alt_A09)
        # Yes, in .ID_ALT
        elif len(indices_alt) > 0:
            foo = np.array(id_alt_A09)[indices_alt][0]
            id_A09 = np.array(id_A09)
            id_A09[indices_alt] = name_K16
            id_A09 = list(id_A09)
        else:
            foo = ''
        output_str = "{0:25s} {1:25s}".format(output_str, foo)

        # Is name_K16 in I15?
        indices = [i for i,x in enumerate(id_I15) if x == name_K16]
        indices_alt = [i for i,x in enumerate(id_alt_I15) if x == name_K16]
        if len(indices) > 0:
            foo = np.array(id_I15)[indices][0]
            id_alt_I15 = np.array(id_alt_I15)
            id_alt_I15[indices] = name_K16
            id_alt_I15 = list(id_alt_I15)
        elif len(indices_alt) > 0:
            foo = np.array(id_alt_I15)[indices_alt][0]
            id_I15 = np.array(id_I15)
            id_I15[indices_alt] = name_K16
            id_I15 = list(id_I15)
        else:
            foo = ''
        output_str = "{0:25s} {1:25s}".format(output_str, foo)

        # Is name_K16 in J17?
        indices = [i for i,x in enumerate(id_J17) if x == name_K16]
        if len(indices) > 0:
            foo = np.array(id_J17)[indices][0]
        else:
            foo = ''
        output_str = "{0:25s} {1:25s}".format(output_str, foo)


        # Add one to source_counter
        source_counter = source_counter + 1
        # Add name_K16 to ID_master
        ID_master.append(name_K16)
        # Add redshift to z_master
        z_master.append(z_K16[i_z])
        output_str = "{0:25s} {1:2.5f}".format(output_str, z_K16[i_z])

        # Output
        if verbose:
            print(output_str)

        i_z = i_z + 1


    # --- Loop through L17
    i_z = 0
    for name_L17 in id_L17:
        output_str = name_L17

        # Is name_L17 in K16?
        indices = [i for i,x in enumerate(id_K16) if x == name_L17]
        if len(indices) == 0:
            foo = ''
            output_str = "{0:4d} {1:25s} {2:25s}".format(source_counter, foo, output_str)

            # Is name_L17 in R15?
            indices = [i for i,x in enumerate(id_R15) if x == name_L17]
            if len(indices) > 0:
                foo = np.array(id_R15)[indices][0]
            else:
                foo = ''
            output_str = "{0:25s} {1:25s}".format(output_str, foo)

            # Is name_L17 in G14?
            indices = [i for i,x in enumerate(id_G14) if x == name_L17]
            if len(indices) > 0:
                foo = np.array(id_G14)[indices][0]
            else:
                foo = ''
            output_str = "{0:25s} {1:25s}".format(output_str, foo)

            # Is name_L17 in A09?
            indices = [i for i,x in enumerate(id_A09) if x == name_L17]
            indices_alt = [i for i,x in enumerate(id_alt_A09) if x == name_L17]
            if len(indices) > 0:
                foo = np.array(id_A09)[indices][0]
                id_alt_A09 = np.array(id_alt_A09)
                id_alt_A09[indices] = name_L17
                id_alt_A09 = list(id_alt_A09)
            elif len(indices_alt) > 0:
                foo = np.array(id_alt_A09)[indices_alt][0]
                id_A09 = np.array(id_A09)
                id_A09[indices_alt] = name_L17
                id_A09 = list(id_A09)
            else:
                foo = ''
            output_str = "{0:25s} {1:25s}".format(output_str, foo)

            # Is name_L17 in I15?
            indices = [i for i,x in enumerate(id_I15) if x == name_L17]
            indices_alt = [i for i,x in enumerate(id_alt_I15) if x == name_L17]
            if len(indices) > 0:
                foo = np.array(id_I15)[indices][0]
                id_alt_I15 = np.array(id_alt_I15)
                id_alt_I15[indices] = name_L17
                id_alt_I15 = list(id_alt_I15)
            elif len(indices_alt) > 0:
                foo = np.array(id_alt_I15)[indices_alt][0]
                id_I15 = np.array(id_I15)
                id_I15[indices_alt] = name_L17
                id_I15 = list(id_I15)
            else:
                foo = ''
            output_str = "{0:25s} {1:25s}".format(output_str, foo)

            # Is name_L17 in J17?
            indices = [i for i,x in enumerate(id_J17) if x == name_L17]
            if len(indices) > 0:
                foo = np.array(id_J17)[indices][0]
            else:
                foo = ''
            output_str = "{0:25s} {1:25s}".format(output_str, foo)

            # Add one to source_counter
            source_counter = source_counter + 1
            # Add name_L17 to ID_master if not in K16
            ID_master.append(name_L17)
            # Add redshift to z_master
            z_master.append(z_L17[i_z])
            output_str = "{0:25s} {1:2.5f}".format(output_str, z_L17[i_z])

            # Output
            if verbose:
                print( output_str)

        i_z = i_z + 1


    # --- Loop through R15
    i_z = 0
    for name_R15 in id_R15:
        output_str = name_R15

        # Is name_R15 in K16 or L17?
        indices_1 = [i for i,x in enumerate(id_K16) if x == name_R15]
        indices_2 = [i for i,x in enumerate(id_L17) if x == name_R15]
        # No
        if len(indices_1) == 0 and len(indices_2) == 0:
            foo = ''
            output_str = "{0:4d} {1:25s} {2:25s} {3:25s}".format(source_counter, foo, foo, output_str)

            # Is name_R15 in G14?
            indices = [i for i,x in enumerate(id_G14) if x == name_R15]
            if len(indices) > 0:
                foo = np.array(id_G14)[indices][0]
            else:
                foo = ''
            output_str = "{0:25s} {1:25s}".format(output_str, foo)

            # Is name_R15 in A09?
            indices = [i for i,x in enumerate(id_A09) if x == name_R15]
            indices_alt = [i for i,x in enumerate(id_alt_A09) if x == name_R15]
            if len(indices) > 0:
                foo = np.array(id_A09)[indices][0]
                id_alt_A09 = np.array(id_alt_A09)
                id_alt_A09[indices] = name_R15
                id_alt_A09 = list(id_alt_A09)
            elif len(indices_alt) > 0:
                foo = np.array(id_alt_A09)[indices_alt][0]
                id_A09 = np.array(id_A09)
                id_A09[indices_alt] = name_R15
                id_A09 = list(id_A09)
            else:
                foo = ''
            output_str = "{0:25s} {1:25s}".format(output_str, foo)

            # Is name_R15 in I15?
            indices = [i for i,x in enumerate(id_I15) if x == name_R15]
            indices_alt = [i for i,x in enumerate(id_alt_I15) if x == name_R15]
            if len(indices) > 0:
                foo = np.array(id_I15)[indices][0]
                id_alt_I15 = np.array(id_alt_I15)
                id_alt_I15[indices] = name_R15
                id_alt_I15 = list(id_alt_I15)
            elif len(indices_alt) > 0:
                foo = np.array(id_alt_I15)[indices_alt][0]
                id_I15 = np.array(id_I15)
                id_I15[indices_alt] = name_R15
                id_I15 = list(id_I15)
            else:
                foo = ''
            output_str = "{0:25s} {1:25s}".format(output_str, foo)

            # Is name_R15 in J17?
            indices = [i for i,x in enumerate(id_J17) if x == name_R15]
            if len(indices) > 0:
                foo = np.array(id_J17)[indices][0]
            else:
                foo = ''
            output_str = "{0:25s} {1:25s}".format(output_str, foo)


            # Add one to source_counter
            source_counter = source_counter + 1
            # Add name_R15 to ID_master if not in K16 nor in L17
            ID_master.append(name_R15)
            # Add redshift to z_master
            z_master.append(z_R15[i_z])
            output_str = "{0:25s} {1:2.5f}".format(output_str, z_R15[i_z])

            # Output
            if verbose:
                print(output_str)

        i_z = i_z + 1


    # --- Loop through G14
    i_z = 0
    for name_G14 in id_G14:
        output_str = name_G14

        # Is name_G14 in K16, L17, or R15?
        indices_1 = [i for i,x in enumerate(id_K16) if x == name_G14]
        indices_2 = [i for i,x in enumerate(id_L17) if x == name_G14]
        indices_3 = [i for i,x in enumerate(id_R15) if x == name_G14]
        if len(indices_1) == 0 and len(indices_2) == 0 and len(indices_3) == 0:
            foo = ''
            output_str = "{0:4d} {1:25s} {2:25s} {3:25s} {4:25s}".format(source_counter, foo, foo, foo, output_str)

            # Is name_G14 in A09?
            indices = [i for i,x in enumerate(id_A09) if x == name_G14]
            indices_alt = [i for i,x in enumerate(id_alt_A09) if x == name_G14]
            if len(indices) > 0:
                foo = np.array(id_A09)[indices][0]
                id_alt_A09 = np.array(id_alt_A09)
                id_alt_A09[indices] = name_K16
                id_alt_A09 = list(id_alt_A09)
            elif len(indices_alt) > 0:
                foo = np.array(id_alt_A09)[indices_alt][0]
                id_A09 = np.array(id_A09)
                id_A09[indices_alt] = name_K16
                id_A09 = list(id_A09)
            else:
                foo = ''
            output_str = "{0:25s} {1:25s}".format(output_str, foo)

            # Is name_G14 in I15?
            indices = [i for i,x in enumerate(id_I15) if x == name_G14]
            indices_alt = [i for i,x in enumerate(id_alt_I15) if x == name_G14]
            if len(indices) > 0:
                foo = np.array(id_I15)[indices][0]
                id_alt_I15 = np.array(id_alt_I15)
                id_alt_I15[indices] = name_G14
                id_alt_I15 = list(id_alt_I15)
            elif len(indices_alt) > 0:
                foo = np.array(id_alt_I15)[indices_alt][0]
                id_I15 = np.array(id_I15)
                id_I15[indices_alt] = name_G14
                id_I15 = list(id_I15)
            else:
                foo = ''
            output_str = "{0:25s} {1:25s}".format(output_str, foo)

            # Is name_G14 in J17?
            indices = [i for i,x in enumerate(id_J17) if x == name_G14]
            if len(indices) > 0:
                foo = np.array(id_J17)[indices][0]
            else:
                foo = ''
            output_str = "{0:25s} {1:25s}".format(output_str, foo)

            # Add one to source_counter
            source_counter = source_counter + 1
            # Add name_R15 to ID_master if not in K16 nor in L17 nor in R15
            ID_master.append(name_G14)
            # Add redshift to z_master
            z_master.append(z_G14[i_z])
            output_str = "{0:25s} {1:2.5f}".format(output_str, z_G14[i_z])

            if verbose:
                print(output_str)

        i_z = i_z + 1

    # --- Loop through A09
    i_z = 0
    for j in np.arange(0,len(table_A09.ID_ALT)):

        name_A09 = table_A09.ID[j]
        name_alt_A09 = table_A09.ID_ALT[j]
        output_str = name_alt_A09

        # Is name_A09 in K16, L17, R15, or G14?
        indices_1 = [i for i,x in enumerate(id_K16) if x == name_A09]
        indices_2 = [i for i,x in enumerate(id_L17) if x == name_A09]
        indices_3 = [i for i,x in enumerate(id_R15) if x == name_A09]
        indices_4 = [i for i,x in enumerate(id_G14) if x == name_A09]

        # Is name_alt_A09 in K16, L17, R15, or G14?
        indices_alt_1 = [i for i,x in enumerate(id_K16) if x == name_alt_A09]
        indices_alt_2 = [i for i,x in enumerate(id_L17) if x == name_alt_A09]
        indices_alt_3 = [i for i,x in enumerate(id_R15) if x == name_alt_A09]
        indices_alt_4 = [i for i,x in enumerate(id_G14) if x == name_alt_A09]

        if len(indices_1) == 0 and len(indices_2) == 0 and len(indices_3) == 0 and len(indices_4) == 0 and len(indices_alt_1) == 0 and len(indices_alt_2) == 0 and len(indices_alt_3) == 0 and len(indices_alt_4) == 0:
            foo = ''
            output_str = "{0:4d} {1:25s} {2:25s} {3:25s} {4:25s} {5:25s}".format(source_counter, foo, foo, foo, foo, output_str)

            # Is name_A09 in I15?
            indices = [i for i,x in enumerate(id_I15) if x == name_A09]
            indices_alt = [i for i,x in enumerate(id_alt_I15) if x == name_A09]
            if len(indices) > 0:
                foo = np.array(id_I15)[indices][0]
                id_alt_I15 = np.array(id_alt_I15)
                id_alt_I15[indices] = name_A09
                id_alt_I15 = list(id_alt_I15)
            elif len(indices_alt) > 0:
                foo = np.array(id_alt_I15)[indices_alt][0]
                id_I15 = np.array(id_I15)
                id_I15[indices_alt] = name_A09
                id_I15 = list(id_I15)
            else:
                foo = ''
            output_str = "{0:25s} {1:25s}".format(output_str, foo)

            # Is name_A09 in J17?
            indices = [i for i,x in enumerate(id_J17) if x == name_A09]
            if len(indices) > 0:
                foo = np.array(id_J17)[indices][0]
            else:
                foo = ''
            output_str = "{0:25s} {1:25s}".format(output_str, foo)

            # Add one to source_counter
            source_counter = source_counter + 1
            # Add name_alt_A09 to ID_master if not in K16 nor in L17 nor in R15 nor in G14
            ID_master.append(name_alt_A09)
            # Add redshift to z_master
            z_master.append(z_A09[i_z])
            output_str = "{0:25s} {1:2.5f}".format(output_str, z_A09[i_z])

            if verbose:
                print(output_str)

        i_z = i_z + 1

    # --- Loop through I15
    i_z = 0
    for j in np.arange(0,len(table_I15.ID_ALT)):

        name_I15 = table_I15.ID[j]
        name_alt_I15 = table_I15.ID_ALT[j]
        output_str = name_alt_I15

        # Is name_A09 in K16, L17, R15, G14, A09 or J17?
        indices_1 = [i for i,x in enumerate(id_K16) if x == name_I15]
        indices_2 = [i for i,x in enumerate(id_L17) if x == name_I15]
        indices_3 = [i for i,x in enumerate(id_R15) if x == name_I15]
        indices_4 = [i for i,x in enumerate(id_G14) if x == name_I15]
        indices_5 = [i for i,x in enumerate(id_A09) if x == name_I15]

        # Is name_alt_A09 in K16, L17, R15, G14, A09 or J17?
        indices_alt_1 = [i for i,x in enumerate(id_K16) if x == name_alt_I15]
        indices_alt_2 = [i for i,x in enumerate(id_L17) if x == name_alt_I15]
        indices_alt_3 = [i for i,x in enumerate(id_R15) if x == name_alt_I15]
        indices_alt_4 = [i for i,x in enumerate(id_G14) if x == name_alt_I15]
        indices_alt_5 = [i for i,x in enumerate(id_A09) if x == name_alt_I15]

        if len(indices_1) == 0 and len(indices_2) == 0 and len(indices_3) == 0 and len(indices_4) == 0 and len(indices_5) == 0 and len(indices_alt_1) == 0 and len(indices_alt_2) == 0 and len(indices_alt_3) == 0 and len(indices_alt_4) == 0 and len(indices_alt_5) == 0:
            foo = ''
            output_str = "{0:4d} {1:25s} {2:25s} {3:25s} {4:25s} {5:25s} {6:25s}".format(source_counter, foo, foo, foo, foo, foo, output_str)

            # Is name_I15 in J17?
            indices = [i for i,x in enumerate(id_J17) if x == name_I15]
            if len(indices) > 0:
                foo = np.array(id_J17)[indices][0]
            else:
                foo = ''
            output_str = "{0:25s} {1:25s}".format(output_str, foo)

            # Add one to source_counter
            source_counter = source_counter + 1
            # Add name_alt_I15 to ID_master if not in K16 nor in L17 nor in R15 nor in G14
            ID_master.append(name_alt_I15)
            # Add redshift to z_master
            z_master.append(z_I15[i_z])
            output_str = "{0:25s} {1:2.5f}".format(output_str, z_I15[i_z])

            if verbose:
                print(output_str)

        i_z = i_z + 1


    # --- Loop through J17
    i_z = 0
    for j in np.arange(0,len(table_J17.ID_ALT)):

        name_J17 = table_J17.ID[j]
        name_alt_J17 = table_J17.ID_ALT[j]
        output_str = name_alt_J17

        # Is name_A09 in K16, L17, R15, G14, A09 or J17?
        indices_1 = [i for i,x in enumerate(id_K16) if x == name_J17]
        indices_2 = [i for i,x in enumerate(id_L17) if x == name_J17]
        indices_3 = [i for i,x in enumerate(id_R15) if x == name_J17]
        indices_4 = [i for i,x in enumerate(id_G14) if x == name_J17]
        indices_5 = [i for i,x in enumerate(id_A09) if x == name_J17]
        indices_6 = [i for i,x in enumerate(id_I15) if x == name_J17]

        # Is name_alt_A09 in K16, L17, R15, G14, A09 or J17?
        indices_alt_1 = [i for i,x in enumerate(id_K16) if x == name_alt_J17]
        indices_alt_2 = [i for i,x in enumerate(id_L17) if x == name_alt_J17]
        indices_alt_3 = [i for i,x in enumerate(id_R15) if x == name_alt_J17]
        indices_alt_4 = [i for i,x in enumerate(id_G14) if x == name_alt_J17]
        indices_alt_5 = [i for i,x in enumerate(id_A09) if x == name_alt_J17]
        indices_alt_6 = [i for i,x in enumerate(id_I15) if x == name_alt_J17]

        if len(indices_1) == 0 and len(indices_2) == 0 and len(indices_3) == 0 and len(indices_4) == 0 and len(indices_5) == 0 and len(indices_6) and len(indices_alt_1) == 0 and len(indices_alt_2) == 0 and len(indices_alt_3) == 0 and len(indices_alt_4) == 0 and len(indices_alt_5) and len(indices_alt_6) == 0:
            foo = ''
            output_str = "{0:4d} {1:25s} {2:25s} {3:25s} {4:25s} {5:25s} {6:25s} {7:25s}".format(source_counter, foo, foo, foo, foo, foo, foo, output_str)

            # Add one to source_counter
            source_counter = source_counter + 1
            # Add name_alt_I15 to ID_master if not in K16 nor in L17 nor in R15 nor in G14
            ID_master.append(name_alt_J17)
            # Add redshift to z_master
            z_master.append(z_J17[i_z])
            output_str = "{0:25s} {1:2.5f}".format(output_str, z_J17[i_z])

            if verbose:
                print(output_str)

        i_z = i_z + 1
    # -------------------------------------------

    # --- At this stage pickle db to avoid slow NED queries ---
    print("")
    print("Master sample size", len(ID_master))
    df = pd.DataFrame({'ID': ID_master, 'z': z_master})
    df.to_pickle(data_path+'master_list.pkl')
    # ---------------------------------------------------------


def commit_to_db_master():
    """ Commits ID_master and z_master to db.
    """
    # --- Read in ID_master list ---
    df = pd.read_pickle(data_path+'master_list.pkl')
    # ------------------------------

    # --- Commit to db ---
    for i in np.arange(0,len(df['ID'])):
        db.local_galaxies.insert_one(
            {
                'ID' : df['ID'][i],
                'z': df['z'][i],
            }
        )
    # --------------------



def commit_to_db_J17():
    """ Commits J17 data to db.
    """

    # --- Read in I15 pkl file ---
    df = pd.read_pickle(data_path+'Jiao-et-al-2017.pkl')
    print(df.SdV_CO10)
    # ----------------------------

    ## --- Commit to db ---
    #for i in np.arange(0,len(df['ID'])):
    #    if int(df['LIR_8_1000'][i]) != ValUnDef:
    #        ID = df['ID'][i]
    #        ID_ALT = df['ID_ALT'][i]

    #        # Check if ID already exists in db
    #        cursor = db.local_galaxies.find_one({'ID': ID})
    #        cursor_alt = db.local_galaxies.find_one({'ID': ID_ALT})

    #        # ID in db['ID']
    #        if not(cursor == None):
    #            db.local_galaxies.update_one(
    #                {
    #                    'ID': {
    #                        "$eq": ID
    #                        }
    #                    },
    #                {'$set': {'LIR_8_1000.J17': df['LIR_8_1000'][i]}
    #                }
    #            )
    #            # Commit LIR_8_1000_I15 to LIR_8_1000.MASTER
    #            db.local_galaxies.update_one(
    #                {
    #                    'ID': {
    #                        "$eq": ID
    #                        }
    #                    },
    #                {'$set': {'LIR_8_1000.MASTER': df['LIR_8_1000'][i]}
    #                }
    #            )
    #        # ID not in db['ID'] and ID_ALT in db['ID'] and ID_ALT != ''
    #        elif cursor == None and not(cursor_alt == None) and ID_ALT != '':
    #            db.local_galaxies.update_one(
    #                {
    #                    'ID': {
    #                        "$eq": ID_ALT
    #                        }
    #                    },
    #                {'$set': {'LIR_8_1000.J17': df['LIR_8_1000'][i]}
    #                }
    #            )
    #            # Commit LIR_8_1000_I15 to LIR_8_1000.MASTER
    #            db.local_galaxies.update_one(
    #                {
    #                    'ID': {
    #                          "$eq": ID_ALT
    #                        }
    #                    },
    #                {'$set': {'LIR_8_1000.MASTER': df['LIR_8_1000'][i]}
    #                }
    #            )
    #        # ID not in db['ID'] and ID_ALT not in db['ID']
    #        else:
    #            print("Error... source should be in db.")
    #    # No LIR measurement
    #    else:
    #        print("Warning... source has no LIR measurement.")
    ## --------------------


    # --- Commit to db ---
    # Transitions in J17
    transitions = ['1-0']
    transitions = ['12CO(' + s + ')' for s in transitions]

    # Line flux entries in df
    entries = ['10']
    entries = ['CO' + s for s in entries]
    entries_err = ['eSdV_' + s for s in entries]
    entries = ['SdV_' + s for s in entries]

    # Loop over I15 pickle files
    for k in ['1']: #, '5']:

        # Read in I15 pkl file
        df = pd.read_pickle(data_path+'Jiao-et-al-2017.pkl')

        for i in np.arange(0,len(df['ID'])):
            ID = df['ID'][i]
            ID_ALT = df['ID_ALT'][i]

            # Loop over line entries for each source
            j=0
            for entry in entries:
                entry_err = entries_err[j]
                # Fixing errors, upper limits and undefined values
                SdV = float(df[entry][i])
                eSdV = float(df[entry_err][i])

                # Check if ID already exists in db
                cursor = db.local_galaxies.find_one({'ID': ID})
                cursor_alt = db.local_galaxies.find_one({'ID': ID_ALT})

                # ID in db['ID'] and ID_ALT not in db['ID']
                if not(cursor == None):
                    # SdV is detected (always the case for I15)
                    if SdV > 0:
                        # Commit to db
                        #SdV = line_flux_conversion(freq[transitions[j]], SdV, conversion='si2jansky')[0]
                        db.local_galaxies.update_one(
                            {
                                'ID': {
                                    "$eq": ID
                                    }
                                },
                            {'$set': {transitions[j]+'.SdV_J17': SdV}
                            }
                        )
                        #eSdV = line_flux_conversion(freq[transitions[j]], eSdV, conversion='si2jansky')[0]
                        db.local_galaxies.update_one(
                            {
                                'ID': {
                                    "$eq": ID
                                    }
                                },
                            {'$set': {transitions[j]+'.eSdV_J17': eSdV}
                            }
                        )
                        # Commit I15 fluxes to master fluxes
                        db.local_galaxies.update_one(
                            {
                                'ID': {
                                    "$eq": ID
                                    }
                                },
                            {'$set': {transitions[j]+'.SdV': SdV}
                            }
                        )
                        db.local_galaxies.update_one(
                            {
                                'ID': {
                                    "$eq": ID
                                    }
                                },
                            {'$set': {transitions[j]+'.eSdV': eSdV}
                            }
                        )

                # ID not in db['ID'] and ID_ALT in db['ID'] and ID_ALT != ''
                elif cursor == None and not(cursor_alt == None) and ID_ALT != '':
                    # SdV is detected (always the case for I15)
                    if SdV > 0:
                        # Commit to db
                        #SdV = line_flux_conversion(freq[transitions[j]], SdV, conversion='si2jansky')[0]
                        db.local_galaxies.update_one(
                            {
                                'ID': {
                                    "$eq": ID_ALT
                                    }
                                },
                            {'$set': {transitions[j]+'.SdV_J17': SdV}
                            }
                        )
                        #eSdV = line_flux_conversion(freq[transitions[j]], eSdV, conversion='si2jansky')[0]
                        db.local_galaxies.update_one(
                            {
                                'ID': {
                                    "$eq": ID_ALT
                                    }
                                },
                            {'$set': {transitions[j]+'.eSdV_J17': eSdV}
                            }
                        )
                        # Commit I15 fluxes to master fluxes
                        db.local_galaxies.update_one(
                            {
                                'ID': {
                                    "$eq": ID_ALT
                                    }
                                },
                            {'$set': {transitions[j]+'.SdV': SdV}
                            }
                        )
                        db.local_galaxies.update_one(
                            {
                                'ID': {
                                    "$eq": ID_ALT
                                    }
                                },
                            {'$set': {transitions[j]+'.eSdV': eSdV}
                            }
                        )

                # ID_ALT not in db['ID_ALT']
                else:
                    print("Error... source should be in db.")

                j=j+1
    # --------------------


def commit_to_db_I15():
    """ Commits I15 data to db.
    """

    # --- Read in I15 pkl file ---
    df = pd.read_pickle(data_path+'Israel-et-al-2015-Table-1.pkl')
    # ----------------------------

    # --- Commit to db ---
    for i in np.arange(0,len(df['ID'])):
        if int(df['LIR_8_1000'][i]) != ValUnDef:
            ID = df['ID'][i]
            ID_ALT = df['ID_ALT'][i]

            # Check if ID already exists in db
            cursor = db.local_galaxies.find_one({'ID': ID})
            cursor_alt = db.local_galaxies.find_one({'ID': ID_ALT})

            # ID in db['ID']
            if not(cursor == None):
                db.local_galaxies.update_one(
                    {
                        'ID': {
                            "$eq": ID
                            }
                        },
                    {'$set': {'LIR_8_1000.I15': df['LIR_8_1000'][i]}
                    }
                )
                # Commit LIR_8_1000_I15 to LIR_8_1000.MASTER
                db.local_galaxies.update_one(
                    {
                        'ID': {
                            "$eq": ID
                            }
                        },
                    {'$set': {'LIR_8_1000.MASTER': df['LIR_8_1000'][i]}
                    }
                )
            # ID not in db['ID'] and ID_ALT in db['ID'] and ID_ALT != ''
            elif cursor == None and not(cursor_alt == None) and ID_ALT != '':
                db.local_galaxies.update_one(
                    {
                        'ID': {
                            "$eq": ID_ALT
                            }
                        },
                    {'$set': {'LIR_8_1000.I15': df['LIR_8_1000'][i]}
                    }
                )
                # Commit LIR_8_1000_I15 to LIR_8_1000.MASTER
                db.local_galaxies.update_one(
                    {
                        'ID': {
                              "$eq": ID_ALT
                            }
                        },
                    {'$set': {'LIR_8_1000.MASTER': df['LIR_8_1000'][i]}
                    }
                )
            # ID not in db['ID'] and ID_ALT not in db['ID']
            else:
                print("Error... source should be in db.")
        # No LIR measurement
        else:
            print("Warning... source has no LIR measurement.")
    # --------------------


    # --- Commit to db ---
    # Transitions in I15
    transitions = ['2-1','4-3','7-6']
    transitions = ['12CO(' + s + ')' for s in transitions]
    transitions = transitions + ['[CI]609','[CI]370']
    transitions = ['13CO(2-1)'] + transitions

    # Line flux entries in df
    entries = ['21', '43','76']
    entries = ['CO' + s for s in entries]
    entries = entries + ['CI609','CI370']
    entries = ['13CO21'] + entries
    entries_err = ['eSdV_' + s for s in entries]
    entries = ['SdV_' + s for s in entries]


    # Loop over I15 pickle files
    for k in ['2','3', '4']: #, '5']:

        # Read in I15 pkl file
        df = pd.read_pickle(data_path+'Israel-et-al-2015-Table-'+k+'.pkl')

        for i in np.arange(0,len(df['ID'])):
            ID = df['ID'][i]
            ID_ALT = df['ID_ALT'][i]

            # Loop over line entries for each source
            j=0
            for entry in entries:
                entry_err = entries_err[j]
                # Fixing errors, upper limits and undefined values
                SdV = float(df[entry][i])
                eSdV = float(df[entry_err][i])

                # Check if ID already exists in db
                cursor = db.local_galaxies.find_one({'ID': ID})
                cursor_alt = db.local_galaxies.find_one({'ID': ID_ALT})

                # ID in db['ID'] and ID_ALT not in db['ID']
                if not(cursor == None):
                    # SdV is detected (always the case for I15)
                    if SdV > 0:
                        # Commit to db
                        SdV = line_flux_conversion(freq[transitions[j]], SdV, conversion='si2jansky')[0]
                        db.local_galaxies.update_one(
                            {
                                'ID': {
                                    "$eq": ID
                                    }
                                },
                            {'$set': {transitions[j]+'.SdV_I15': SdV}
                            }
                        )
                        eSdV = line_flux_conversion(freq[transitions[j]], eSdV, conversion='si2jansky')[0]
                        db.local_galaxies.update_one(
                            {
                                'ID': {
                                    "$eq": ID
                                    }
                                },
                            {'$set': {transitions[j]+'.eSdV_I15': eSdV}
                            }
                        )
                        # Commit I15 fluxes to master fluxes
                        db.local_galaxies.update_one(
                            {
                                'ID': {
                                    "$eq": ID
                                    }
                                },
                            {'$set': {transitions[j]+'.SdV': SdV}
                            }
                        )
                        db.local_galaxies.update_one(
                            {
                                'ID': {
                                    "$eq": ID
                                    }
                                },
                            {'$set': {transitions[j]+'.eSdV': eSdV}
                            }
                        )

                # ID not in db['ID'] and ID_ALT in db['ID'] and ID_ALT != ''
                elif cursor == None and not(cursor_alt == None) and ID_ALT != '':
                    # SdV is detected (always the case for I15)
                    if SdV > 0:
                        # Commit to db
                        SdV = line_flux_conversion(freq[transitions[j]], SdV, conversion='si2jansky')[0]
                        db.local_galaxies.update_one(
                            {
                                'ID': {
                                    "$eq": ID_ALT
                                    }
                                },
                            {'$set': {transitions[j]+'.SdV_I15': SdV}
                            }
                        )
                        eSdV = line_flux_conversion(freq[transitions[j]], eSdV, conversion='si2jansky')[0]
                        db.local_galaxies.update_one(
                            {
                                'ID': {
                                    "$eq": ID_ALT
                                    }
                                },
                            {'$set': {transitions[j]+'.eSdV_I15': eSdV}
                            }
                        )
                        # Commit I15 fluxes to master fluxes
                        db.local_galaxies.update_one(
                            {
                                'ID': {
                                    "$eq": ID_ALT
                                    }
                                },
                            {'$set': {transitions[j]+'.SdV': SdV}
                            }
                        )
                        db.local_galaxies.update_one(
                            {
                                'ID': {
                                    "$eq": ID_ALT
                                    }
                                },
                            {'$set': {transitions[j]+'.eSdV': eSdV}
                            }
                        )

                # ID_ALT not in db['ID_ALT']
                else:
                    print("Error... source should be in db.")

                j=j+1
    # --------------------




def commit_to_db_A09():
    """ Commits A09 data to db.
    """
    # --- Read in A09 pkl file ---
    df = pd.read_pickle(data_path+'Armus-et-al-2009.pkl')
    # ----------------------------

    # --- Commit to db ---
    for i in np.arange(0,len(df['ID'])):
        # Only proceed if LIR measurement exist
        if int(df['LIR_8_1000'][i]) != ValUnDef:
            ID = df['ID'][i]
            ID_ALT = df['ID_ALT'][i]

            # Check if ID and ID_ALT already exists in db['ID']
            cursor = db.local_galaxies.find_one({'ID': ID})
            cursor_alt = db.local_galaxies.find_one({'ID': ID_ALT})

            # ID in db['ID'] and ID_ALT no in db['ID']
            if not(cursor == None):
                db.local_galaxies.update_one(
                    {
                        'ID': {
                            "$eq": ID
                            }
                        },
                    {'$set': {'LIR_8_1000.A09': df['LIR_8_1000'][i]}
                    }
                )
                # Commit LIR_8_1000_A09 to LIR_8_1000.MASTER
                db.local_galaxies.update_one(
                    {
                        'ID': {
                            "$eq": ID
                            }
                        },
                    {'$set': {'LIR_8_1000.MASTER': df['LIR_8_1000'][i]}
                    }
                )
            # ID not in db['ID'] and ID_ALT in db['ID']
            elif cursor == None and not(cursor_alt == None):
                db.local_galaxies.update_one(
                    {
                        'ID': {
                            "$eq": ID_ALT
                            }
                        },
                    {'$set': {'LIR_8_1000.A09': df['LIR_8_1000'][i]}
                    }
                )
                # Commit LIR_8_1000_A09 to LIR_8_1000.MASTER
                db.local_galaxies.update_one(
                    {
                        'ID': {
                            "$eq": ID_ALT
                            }
                        },
                    {'$set': {'LIR_8_1000.MASTER': df['LIR_8_1000'][i]}
                    }
                )
            # ID and ID_ALT not in db['ID']: Error
            else:
                print("Error... source should be in db.")
        # No LIR measurement
        else:
            print("Warning... source has no LIR measurement.")
    # --------------------



def commit_to_db_G14():
    """ Commits G14 data to db.
    """
    # --- Read in G14 pkl file ---
    df = pd.read_pickle(data_path+'Greve-et-al-2014.pkl')
    # ----------------------------

    # --- Commit to db ---
    for i in np.arange(0,len(df['ID'])):
        # Only proceed if LIR measurement exist
        if df['LIR_8_1000'][i] != ValUnDef:
            ID = df['ID'][i]

            # Check if ID already exists in db['ID']
            cursor = db.local_galaxies.find_one({'ID': ID})

            # ID in db['ID']
            if not(cursor == None):
                db.local_galaxies.update_one(
                    {
                        'ID': {
                            "$eq": ID
                            }
                        },
                    {'$set': {'LIR_8_1000.G14': df['LIR_8_1000'][i]}
                    }
                )
                db.local_galaxies.update_one(
                    {
                        'ID': {
                            "$eq": ID
                            }
                        },
                    {'$set': {'LIR_50_300.G14': df['LIR_50_300'][i]}
                    }
                )
                # Commit LIR_8_1000_G14 to LIR_8_1000.MASTER
                db.local_galaxies.update_one(
                    {
                        'ID': {
                            "$eq": ID
                            }
                        },
                    {'$set': {'LIR_8_1000.MASTER': df['LIR_8_1000'][i]}
                    }
                )
            # ID not in db['ID']
            else:
                print("Error... source should be in db.")
        # No LIR measurement
        else:
            print("Warning... source has no LIR measurement.")
    # --------------------


def _fix_line_fluxes_R15(SdV, eF, j):

    if j < 3:
        if not('...' in SdV):
            # [W m^-2]
            SdV = float(SdV)*1.E-18
            eSdV = eF*SdV
        else:
            SdV = ValUnDef
            eSdV = ValUnDef
    else:
        if '^a' in SdV:
            SdV = SdV.replace('^a','')
            indices_start = SdV.find('(')
            indices_end = SdV.find(')')
            SdV = float(SdV[indices_start+1:indices_end])
            # [W m^-2]
            SdV = SdV*1.E-17
            eSdV = 0.16*SdV
        elif '-' in SdV:
            SdV = ValUnDef
            eSdV = ValUnDef
        elif not('^a' in SdV) and '(' in SdV:
            indices_end = SdV.find('(')
            SdV = float(SdV[0:indices_end])
            # [W m^-2]
            SdV = SdV*1.E-17
            eSdV = 0.16*SdV
        elif '^b' in SdV:
            eSdV = ValUpLim
            SdV = float(SdV.replace('^b',''))
            # [W m^-2]
            SdV = SdV*1.E-17
        elif '^d' in SdV:
            eSdV = ValUnDef
            SdV = ValUnDef
        else:
            SdV = float(SdV)
            SdV = SdV*1.E-17
            eSdV = 0.16*SdV

    # Set negative fluxes to ValUnDef
    if SdV < 0:
        SdV = ValUnDef
        eSdV = ValUnDef

    return SdV, eSdV


def commit_to_db_R15():
    """ Commits R15 data to db.
    """
    # Adopted flux uncertainty (low-J CO lines)
    eF = 0.30

    # --- Read in R15 pickle pkl file ---
    df = pd.read_pickle(data_path+'Rosenberg-et-al-2015.pkl')
    # -----------------------------------

    # Transitions in R15
    transitions = ['1-0', '2-1', '3-2', '4-3','5-4','6-5','7-6','8-7','9-8','10-9','11-10','12-11','13-12']
    transitions = ['12CO(' + s + ')' for s in transitions]
    transitions = transitions + ['[CI]609','[CI]370','[OI]63','[OI]145','[CII]158']


    # Line flux entries in R15
    entries = ['10','21','32','43','54','65','76','87','98','109','1110','1211','1312']
    entries = ['CO' + s for s in entries]
    entries = entries + ['CI609','CI370','OI63', 'OI145', 'CII158']
    entries = ['SdV_' + s for s in entries]

    # --- Commit to db ---
    for i in np.arange(0,len(df['ID'])):
        ID = df['ID'][i]

        # Check if ID already exists in db['ID']
        cursor = db.local_galaxies.find_one({'ID': ID})

        # ID in db['ID']
        if not(cursor == None):
            if int(df['LIR_8_1000'][i]) != ValUnDef:
                # Commit IR luminosities to db
                db.local_galaxies.update_one(
                    {
                        'ID': {
                            "$eq": ID
                            }
                    },
                    {'$set': {'LIR_8_1000.R15': df['LIR_8_1000'][i]}
                    }
                )
                # Commit LIR_8_1000_R15 to LIR_8_1000.MASTER
                db.local_galaxies.update_one(
                    {
                        'ID': {
                            "$eq": ID
                            }
                        },
                    {'$set': {'LIR_8_1000.MASTER': df['LIR_8_1000'][i]}
                    }
                )
            # No LIR measurement
            else:
                print("Warning... source has no LIR measurement.")
            # Loop over line entries for each source
            j=0
            for entry in entries:
                SdV = str(df[entry][i])
                # Fixing errors, upper limits and undefined values
                SdV, eSdV = _fix_line_fluxes_R15(SdV, eF, j)

                if SdV != ValUnDef and eSdV != ValUpLim:
                    SdV = line_flux_conversion(freq[transitions[j]], SdV, conversion='si2jansky')[0]
                    # Commit line fluxes to db
                    db.local_galaxies.update_one(
                        {
                            'ID': {
                                "$eq": ID
                            }
                        },
                        {'$set': {transitions[j]+'.SdV_R15': SdV}
                        }
                    )
                    eSdV = line_flux_conversion(freq[transitions[j]], eSdV, conversion='si2jansky')[0]
                    # Commit line flux errors to db
                    db.local_galaxies.update_one(
                        {
                            'ID': {
                                "$eq": ID
                            }
                        },
                        {'$set': {transitions[j]+'.eSdV_R15': eSdV}
                        }
                    )
                    # Commit R15 fluxes to master fluxes
                    db.local_galaxies.update_one(
                        {
                            'ID': {
                                "$eq": ID
                                }
                            },
                        {'$set': {transitions[j]+'.SdV': SdV}
                        }
                    )
                    db.local_galaxies.update_one(
                        {
                            'ID': {
                                "$eq": ID
                                }
                            },
                        {'$set': {transitions[j]+'.eSdV': SdV}
                        }
                    )
                elif eSdV == ValUpLim:
                    SdV = line_flux_conversion(freq[transitions[j]], SdV, conversion='si2jansky')[0]
                    # Commit line fluxe errors to db
                    db.local_galaxies.update_one(
                        {
                            'ID': {
                                "$eq": ID
                            }
                        },
                        {'$set': {transitions[j]+'.SdV_3sigmaUL_R15': SdV}
                        }
                    )
                    # Commit R15 fluxes to master fluxes
                    db.local_galaxies.update_one(
                        {
                            'ID': {
                                "$eq": ID
                                }
                            },
                        {'$set': {transitions[j]+'.SdV': SdV}
                        }
                    )
                    db.local_galaxies.update_one(
                        {
                            'ID': {
                                "$eq": ID
                                }
                            },
                        {'$set': {transitions[j]+'.eSdV': eSdV}
                        }
                    )

                j = j+1
        # ID not in db['ID']
        else:
            print("Error... source should be in db.")
    # --------------------


def commit_to_db_K16():
    """ Commits K16 data to db.
    """

    # --- Read in K16 pkl file ---
    df = pd.read_pickle(data_path+'Kamenetzky-et-al-2016.pkl')
    # ----------------------------

    transitions = ['1-0','2-1','3-2','4-3','5-4','6-5','7-6','8-7','9-8',
            '10-9','11-10','12-11', '13-12']
    transitions = ['12CO(' + s + ')' for s in transitions]
    transitions = transitions + ['[CI]609','[CI]370','[NII]205']

    entries = ['10','21','32','43','54','65','76','87','98','109','1110',
            '1211','1312']
    entries = ['CO' + s for s in entries]
    entries = entries + ['CI609','CI370','NII205']

    # --- Commit to db ---
    for i in np.arange(0,len(df['ID'])):
        ID = df['ID'][i]
        LIR = df['LIR_40_120'][i]

        # Check if ID already exists in db
        cursor = db.local_galaxies.find_one({'ID': ID})

        # ID in db['ID']
        if not(cursor == None):

            # Commit LIR if defined
            if LIR != ValUnDef:
                db.local_galaxies.update_one(
                    {
                        'ID': {
                            "$eq": ID
                            }
                        },
                    {'$set': {'LIR_40_120.K16': LIR}
                    }
                )
                # Commit LIR_8_1000_K16 to LIR_8_1000.MASTER
                db.local_galaxies.update_one(
                    {
                        'ID': {
                            "$eq": ID
                            }
                        },
                    {'$set': {'LIR_8_1000.MASTER': LIR}
                    }
                )
            # No LIR measurement
            else:
                print("Warning... source has no LIR measurement.")

            # Commit transitions if defined
            j=0
            for transition in transitions:
                SdV = float(df['SdV_'+entries[j]][i])
                eSdV = float(df['eSdV_'+entries[j]][i])
                SdV_3sigmaUL = float(df['SdV_3sigmaUL_' + entries[j]][i])
                # SdV is defined
                if SdV != ValUnDef:
                    db.local_galaxies.update_one(
                        {
                            'ID': {
                                "$eq": ID
                                }
                            },
                        {'$set': {transition+'.SdV_K16': SdV}
                        }
                    )
                    # Commit K16 fluxes to master fluxes
                    db.local_galaxies.update_one(
                        {
                            'ID': {
                                "$eq": ID
                                }
                            },
                        {'$set': {transition+'.SdV': SdV}
                        }
                    )
                # eSdV is defined
                if eSdV != ValUnDef:
                    db.local_galaxies.update_one(
                        {
                            'ID': {
                                "$eq": ID
                                }
                            },
                        {'$set': {transition+'.eSdV_K16': eSdV}
                        }
                    )
                    # Commit K16 fluxes to master fluxes
                    db.local_galaxies.update_one(
                        {
                            'ID': {
                                "$eq": ID
                                }
                            },
                        {'$set': {transition+'.eSdV': eSdV}
                        }
                    )
                # SdV_3sigmaUL is defined
                if (SdV_3sigmaUL != ValUnDef):
                    db.local_galaxies.update_one(
                        {
                            'ID': {
                                "$eq": ID
                                }
                            },
                        {'$set': {transition+'.SdV_3sigmaUL_K16': SdV_3sigmaUL}
                        }
                    )
                    # Set SdV and eSdV for master fluxes in case of upper limit
                    SdV = SdV_3sigmaUL
                    eSdV = ValUpLim

                    # Commit K16 fluxes to master fluxes
                    db.local_galaxies.update_one(
                        {
                            'ID': {
                                "$eq": ID
                                }
                            },
                        {'$set': {transition+'.SdV': SdV}
                        }
                    )
                    db.local_galaxies.update_one(
                        {
                            'ID': {
                                "$eq": ID
                                }
                            },
                        {'$set': {transition+'.eSdV': eSdV}
                        }
                    )

                j = j+1

        # ERROR: ID not in db['ID']
        else:
            print("Error... source should be in db.")
    # --------------------


def commit_to_db_L17():
    """ Commits L17 data to db.
    """

    # --- Read in L17 pkl file ---
    df = pd.read_pickle(data_path+'Lu-et-al-2017.pkl')
    # ----------------------------

    # Transitions in L17
    transitions = ['4-3','5-4','6-5','7-6','8-7','9-8','10-9','11-10','12-11',
                   '13-12']
    transitions = ['12CO(' + s + ')' for s in transitions]
    transitions = transitions + ['[CI]609','[CI]370','[NII]205']

    # Line flux entries in df
    entries = ['43','54','65','76','87','98','109','1110','1211','1312']
    entries = ['CO' + s for s in entries]
    entries = entries + ['CI609','CI370','NII205']
    entries_err = ['eSdV_' + s for s in entries]
    entries = ['SdV_' + s for s in entries]


    # --- Commit to db ---
    for i in np.arange(0,len(df['ID'])):
        ID = df['ID'][i]
        LIR = df['LIR_8_1000'][i]

        if int(LIR) != ValUnDef:
            # Check if ID already exists in db['ID']
            cursor = db.local_galaxies.find_one({'ID': ID})

            # ID in db['ID']
            if not(cursor == None):
                db.local_galaxies.update_one(
                    {
                        'ID': {
                            "$eq": ID
                            }
                        },
                    {'$set': {'LIR_8_1000.L17': df['LIR_8_1000'][i]}
                    }
                )
                # Commit LIR_8_1000_L17 to LIR_8_1000.MASTER
                db.local_galaxies.update_one(
                    {
                        'ID': {
                            "$eq": ID
                            }
                        },
                    {'$set': {'LIR_8_1000.MASTER': df['LIR_8_1000'][i]}
                    }
                )
            # ID not in db['ID']
            else:
                print("Error... source should be in db.")
        # No LIR measurement
        else:
            print("Warning... source has no LIR measurement.")
    # --------------------

    # --- Commit to db ---
    for i in np.arange(0,len(df['ID'])):
        ID = df['ID'][i]

        # Loop over line entries for each source
        j=0
        for entry in entries:
            entry_err = entries_err[j]
            # Fixing errors, upper limits and undefined values
            SdV = float(df[entry][i])
            eSdV = float(df[entry_err][i])

            # Check if ID already exists in db['ID'] or db['ID_ALT']
            cursor = db.local_galaxies.find_one({'ID': ID})

            # ID in db['ID']
            if not(cursor == None):
                # SdV is detected
                if SdV > 0:
                    # Convert fluxes from [W m^-2] to [Jy km/s]
                    SdV = SdV*1.E-17
                    SdV = line_flux_conversion(freq[transitions[j]], SdV, conversion='si2jansky')[0]
                    eSdV = eSdV*1.E-17
                    eSdV = line_flux_conversion(freq[transitions[j]], eSdV, conversion='si2jansky')[0]
                    # Commit to db
                    db.local_galaxies.update_one(
                        {
                            'ID': {
                                "$eq": ID
                                }
                            },
                        {'$set': {transitions[j]+'.SdV_L17': SdV}
                        }
                    )
                    db.local_galaxies.update_one(
                        {
                            'ID': {
                                "$eq": ID
                                }
                            },
                        {'$set': {transitions[j]+'.eSdV_L17': eSdV}
                        }
                    )
                # Upper limit
                elif SdV < 0 and SdV > -90:
                    # Convert fluxes from [W m^-2] to [Jy km/s]
                    SdV_3sigmaUL = abs(SdV)*1.E-17
                    SdV_3sigmaUL = line_flux_conversion(freq[transitions[j]], SdV_3sigmaUL, conversion='si2jansky')[0]
                    # Commit to db
                    db.local_galaxies.update_one(
                        {
                            'ID': {
                                "$eq": ID
                                }
                            },
                        {'$set': {transitions[j]+'.SdV_3sigmaUL_L17': SdV_3sigmaUL}
                        }
                    )
                    # Set SdV and eSdV for master fluxes
                    SdV = SdV_3sigmaUL
                    eSdV = ValUpLim

                # Commit L17 fluxes to master fluxes
                db.local_galaxies.update_one(
                    {
                        'ID': {
                            "$eq": ID
                            }
                        },
                    {'$set': {transitions[j]+'.SdV': SdV}
                    }
                )
                db.local_galaxies.update_one(
                    {
                        'ID': {
                            "$eq": ID
                            }
                        },
                    {'$set': {transitions[j]+'.eSdV': eSdV}
                    }
                )
            # ID not in db['ID']
            else:
                print("Error... source should be in db.")
            j=j+1
    # --------------------



def extract_source_from_db(ID):
    """ Extract entry for a single source
    """

    cursor = db.local_galaxies.find({'ID': {'$eq': ID}})
    # ID not in db['ID']
    if cursor == None:
        cursor = db.local_galaxies.find({'ID_ALT': {'$eq': ID}})

    return cursor


def _progressBar(preample, value, endvalue, bar_length=20):
    """ Outputs progress bar to terminal.
    """

    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length)-1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    sys.stdout.write("\r"+preample+"Percent: [{0}] {1}%         ".format(arrow + spaces,
                                                                int(round(percent * 100))))
    sys.stdout.flush()
