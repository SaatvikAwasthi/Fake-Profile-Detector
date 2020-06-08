from database import mysql_connector
import re


class dataFactory():
    __dc = ""
    __name_pattern = '^([a-z|A-Z]){1,200}$'
    __email_pattern = '^([A-Z|a-z|0-9](\.|_){0,1})+[A-Z|a-z|0-9]\@([A-Z|a-z|0-9])+((\.){0,1}[A-Z|a-z|0-9]){2}\.[a-z]{2,3}$'
    __number_pattern = '^([0-9]){1,3}$'
    __gender_pattern = '^[(MmFfOo){1}]$'

    def __init__(self):
        self.__dc = mysql_connector.databaseConnector()

    def validName(self, input):
        result = re.match(self.__name_pattern, input)
        if result:
            return input
        else:
            return False

    def validNumber(self, input):
        result = re.match(self.__number_pattern, input)
        if result:
            return input
        else:
            return False

    def validEmail(self, input):
        result = re.match(self.__email_pattern, input)
        if result:
            return input
        else:
            return False

    def validGender(self, input):
        result = re.match(self.__gender_pattern, input)
        if result:
            return input
        else:
            return False

    def registerUser(self, firstname, lastname, age, gender, email, osn_id):
        self.__dc.connectDb()
        name = firstname + " " + lastname
        if self.__dc.insertToDb('user_profile', [name, email]):
            user_id = self.__dc.getUserId(email)
            if(user_id != -9999):
                if self.__dc.insertToDb('user_data', [user_id, age, gender, firstname, lastname]):
                    if self.__dc.insertToDb('user_osn', [user_id, osn_id]):
                        result = user_id
                    else:
                        result = "Error"
                else:
                    result = "Error"
            else:
                result = "Error"
        else:
            result = "Error"

        self.__dc.disconnectDb()

        return result

    def fakeAccountAttempt(self, user_id, osn):
        self.__dc.connectDb()
        user_name = self.__dc.getUserName(user_id)
        if user_name != "-9999":
            if self.__dc.insertToDb('fake_profile_attempt', [user_id, osn]):
                result = user_name
            else:
                result = "Error"
        else:
            result = "Error"

        self.__dc.disconnectDb()

        return result
