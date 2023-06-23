import shutil
import sqlite3
from datetime import datetime
from os import listdir
import os
import csv
from application_logging.logger import App_Logger



class dBOperation:
    """
      This class shall be used for handling all the SQL operations.

      Version: 1.0
      Revisions: None

    """
    def __init__(self):
        self.path = 'Training_Database/'
        self.badFilePath = "Training_Raw_files_validated/Bad_Raw"
        self.goodFilePath = "Training_Raw_files_validated/Good_Raw"
        self.logger = App_Logger()


    def dataBaseConnection(self,DatabaseName):

        """
            Method Name: dataBaseConnection
            Description: This method creates the database with the given name and if Database already exists then opens the connection to the DB.
            Output: Connection to the DB
            On Failure: Raise ConnectionError

            Version: 1.0
            Revisions: None

        """
        try:
            conn = sqlite3.connect(self.path+DatabaseName+'.db')

            file = open("Training_Logs/DataBaseConnectionLog.txt", 'a+')
            self.logger.log(file, "Opened %s database successfully" % DatabaseName)
            file.close()
        except ConnectionError:
            file = open("Training_Logs/DataBaseConnectionLog.txt", 'a+')
            self.logger.log(file, "Error while connecting to database: %s" %ConnectionError)
            file.close()
            raise ConnectionError
        return conn

    def createTableDb(self,DatabaseName,column_names):
        """
            Method Name: createTableDb
            Description: This method creates a table in the given database which will be used to insert the Good data after raw data validation.
            Output: None
            On Failure: Raise Exception

            Version: 1.0
            Revisions: None
        """
        try:
            conn = self.dataBaseConnection(DatabaseName)
            c=conn.cursor()
            c.execute("SELECT count(name)  FROM sqlite_master WHERE type = 'table' AND name = 'Good_Raw_Data'")
            if c.fetchone()[0] ==1:             # checking if the table exists
                conn.close()
                file = open("Training_Logs/DbTableCreateLog.txt", 'a+')
                self.logger.log(file, "Tables created successfully!!")
                file.close()

                file = open("Training_Logs/DataBaseConnectionLog.txt", 'a+')
                self.logger.log(file, "Closed %s database successfully" % DatabaseName)
                file.close()

            else:

                # for key in column_names.keys():
                #     type = column_names[key]
                self.columns = column_names
                self.column_names = list(self.columns.keys())
                self.type = [self.columns[i] for i in self.column_names]
                    # In below try block, we check if the table exists, if yes then add columns to the table
                    # else in catch block we will create the table
                    # try:
                        # c.execute('ALTER TABLE Good_Raw_Data ADD COLUMN "{column_name}" {dataType}'.format(column_name=key,dataType=type))
                    # except:
                try:
                    c.execute('ALTER TABLE Good_Raw_Data ADD COLUMN "{column_name}" {dataType}').format(column_name=self.column_names, dataType=self.type)
                except:
                    c.execute('CREATE TABLE  Good_Raw_Data ({column_name} {dataType})'.format(column_name=self.column_names, dataType=self.type))
                    # c.execute('CREATE TABLE  Good_Raw_Data "{column_name}" {dataType}').format(column_name=self.column_names, dataType=self.type))

                conn.close()

                file = open("Training_Logs/DbTableCreateLog.txt", 'a+')
                self.logger.log(file, "Tables created successfully!!")
                file.close()

                file = open("Training_Logs/DataBaseConnectionLog.txt", 'a+')
                self.logger.log(file, "Closed %s database successfully" % DatabaseName)
                file.close()
            # conn.close()

        except Exception as e:
            file = open("Training_Logs/DbTableCreateLog.txt", 'a+')
            self.logger.log(file, "Error while creating table: %s " % e)
            file.close()
            conn.close()
            file = open("Training_Logs/DataBaseConnectionLog.txt", 'a+')
            self.logger.log(file, "Closed %s database successfully" % DatabaseName)
            file.close()
            raise e


    def insertIntoTableGoodData(self,Database,column_names):

        """
            Method Name: insertIntoTableGoodData
            Description: This method inserts the Good data files from the Good_Raw folder into the above created table.
            Output: None
            On Failure: Raise Exception

            Version: 1.0
            Revisions: None

        """

        conn = self.dataBaseConnection(Database)
        self.createTableDb('Training', column_names)  # added 23 june
        c = conn.cursor()
        # self.column_names = [i for i in column_names.keys()]
        self.column_names = column_names
        goodFilePath= self.goodFilePath
        badFilePath = self.badFilePath
        onlyfiles = [f for f in listdir(goodFilePath)]
        log_file = open("Training_Logs/DbInsertLog.txt", 'a+')

        for file in onlyfiles:
            try:
                with open(goodFilePath+'/'+file, "r") as f:
                    next(f)
                    reader = csv.reader(f, delimiter="\n")
                    self.logger.log(log_file," %s: File loaded successfully till line 142 DataTypeValidation!!" % file)   # adding it to check
                    for line in enumerate(reader):
                        self.logger.log(log_file," %s: File loaded successfully till line 144 DataTypeValidation!!" % file)   # adding it to check
                        for list_ in (line[1]):
                            self.logger.log(log_file," %s: File loaded successfully till line 146 DataTypeValidation!!" % file)   # adding it to check
                            try:
                                c.execute('INSERT INTO Good_Raw_Data {column_names} values {values}'.format(column_names=(self.column_names),values=(list_)))
                                self.logger.log(log_file," %s: File loaded successfully till line 149 DataTypeValidation!!" % file)   # adding it to check
                                # c.execute('INSERT INTO Good_Raw_Data values ({values})'.format(values=list_))
                                # c.execute('INSERT INTO Good_Raw_Data ({column_names}) values ({values})'.format(column_names=(column_names),values=(list_)))
                                self.logger.log(log_file," %s: File loaded successfully!!" % file)
                                conn.commit()
                                self.logger.log(log_file," %s: conn.commit() happened line 154 DataTypeValidation!!" % file)   # adding it to check
                            except Exception as e:
                                raise e

            except Exception as e:

                conn.rollback()
                self.logger.log(log_file,"Error while creating table: %s " % e)
                shutil.move(goodFilePath+'/' + file, badFilePath)
                self.logger.log(log_file, "File Moved Successfully %s" % file)
                log_file.close()
                # conn.close()

        conn.close()
        log_file.close()


    def selectingDatafromtableintocsv(self,Database):

        """
            Method Name: selectingDatafromtableintocsv
            Description: This method exports the data in GoodData table as a CSV file in a given location.
            Output: None
            On Failure: Raise Exception

            Version: 1.0
            Revisions: None

        """

        self.fileFromDb = 'Training_FileFromDB/'
        self.fileName = 'InputFile.csv'
        log_file = open("Training_Logs/ExportToCsv.txt", 'a+')
        try:
            conn = self.dataBaseConnection(Database)
            sqlSelect = "SELECT *  FROM Good_Raw_Data"
            cursor = conn.cursor()

            cursor.execute(sqlSelect)

            results = cursor.fetchall()
            # Get the headers of the csv file
            headers = [i[0] for i in cursor.description]

            #Make the CSV ouput directory
            if not os.path.isdir(self.fileFromDb):
                os.makedirs(self.fileFromDb)

            # Open CSV file for writing.
            csvFile = csv.writer(open(self.fileFromDb + self.fileName, 'w', newline=''),delimiter=',', lineterminator='\r\n',quoting=csv.QUOTE_ALL, escapechar='\\')

            # Add the headers and data to the CSV file.
            csvFile.writerow(headers)
            csvFile.writerows(results)

            self.logger.log(log_file, "File exported successfully!!!")
            log_file.close()

        except Exception as e:
            self.logger.log(log_file, "File exporting failed. Error : %s" %e)
            log_file.close()





