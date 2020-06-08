import pymysql as pms


class databaseConnector():
    # vars
    __host = 'localhost'
    __port = 3306
    __user = 'root'
    __passwd = '123'
    __database = 'fakeprofile'
    __tables = ['fake_profile_attempt',
                'user_data', 'user_osn', 'user_profile']
    __conn = ""
    __INIT_CHECK = False

    def __init__(self):
        self.connectDb()
        cur = self.__conn.cursor()
        cur.execute("SHOW DATABASES")
        if self.__database in cur:
            cur.execute("SHOW TABLES")
            for table in self.__tables:
                if not (table in cur):
                    self.__INIT_CHECK = False
                    break
        else:
            self.__INIT_CHECK = False
        self.disconnectDb()

    def connectDb(self):
        self.__conn = pms.connect(host=self.__host, port=self.__port,
                                  user=self.__user, passwd=self.__passwd, db=self.__database)

    def disconnectDb(self):
        self.__conn.close()

    def insertToDb(self, table, values):
        insert_query = {
            'fake_profile_attempt': 'INSERT INTO `fake_profile_attempt`(`user_id`, `osn_platform`) VALUES (%s,%s)',
            'user_data': 'INSERT INTO `user_data`(`user_id`, `age`, `gender`, `firstname`, `lastname`) VALUES (%s,%s,%s,%s,%s)',
            'user_osn': 'INSERT INTO `user_osn`(`user_id`, `DemoOSN`) VALUES (%s,%s)',
            'user_profile': 'INSERT INTO `user_profile`(`name`, `email`) VALUES (%s,%s)'
        }
        insert_values = ""
        if table == 'fake_profile_attempt':
            insert_values = (values[0], values[1])
        elif table == 'user_data':
            insert_values = (values[0], values[1],
                             values[2], values[3], values[4])
        elif table == 'user_osn':
            insert_values = (values[0], values[1])
        elif table == 'user_profile':
            insert_values = (values[0], values[1])
        else:
            return "Table Not Found"

        cur = self.__conn.cursor()
        sql = insert_query[table]
        val = insert_values
        cur.execute(sql, val)
        self.__conn.commit()

        count = cur.rowcount
        cur.close()

        if(count == 1):
            return True
        else:
            return False

    def readFromDb(self, query):
        cur = self.__conn.cursor()
        cur.execute(query)

        count = cur.rowcount

        if(count > 0):
            result = cur.fetchall()
            return True, result
        else:
            return False

    def getUserId(self, email):
        cur = self.__conn.cursor()
        sql = "SELECT user_id FROM user_profile WHERE email = '" + email + "'"
        cur.execute(sql)

        count = cur.rowcount

        if(count == 1):
            result = cur.fetchall()
            return result[0][0]
        else:
            return -9999

    def getUserName(self, user_id):
        cur = self.__conn.cursor()
        sql = "SELECT name FROM user_profile WHERE user_id = " + user_id
        cur.execute(sql)

        count = cur.rowcount

        if(count == 1):
            result = cur.fetchall()
            return result[0][0]
        else:
            return "-9999"
