import mysql.connector
import pandas as pd

class RemoteServer:
    def __init__(self, args):
        self.host = args.host 
        self.username = args.username 
        self.password = args.password 
        self.database = args.database

    def run_sql_query(self, query):
        con = mysql.connector.connect(host = self.host, user = self.username, password = self.password, database = self.database)
        df = pd.read_sql(query, con)
        return df

def initialize_server(args):
    server = RemoteServer(args=args)
    return server

def get_sql_data(server, query):
    return server.run_sql_query(query=query)