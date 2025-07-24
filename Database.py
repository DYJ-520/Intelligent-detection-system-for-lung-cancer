import sqlite3
import os
from datetime import datetime

# 数据库配置
DB_DIR = 'data-unversioned/db'
DB_PATH = os.path.join(DB_DIR, 'ct_paths.db')

# 确保数据库目录存在
os.makedirs(DB_DIR, exist_ok=True)

def setup_database():
    """初始化数据库表结构，把series_uid当成主键"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS ct_data (
                    series_uid TEXT PRIMARY KEY,    
                    mhd_path TEXT NOT NULL,
                    raw_path TEXT,
                    upload_date TEXT NOT NULL
                )
            ''')
        print(f"数据库初始化成功: {DB_PATH}")
    except sqlite3.Error as e:
        print(f"数据库初始化失败: {e}")

def insert_ct_record(series_uid, mhd_path, raw_path=None):
    """插入CT数据记录"""
    try:
        upload_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  #获取现在的时间
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute(
                'INSERT OR REPLACE INTO ct_data (series_uid, mhd_path, raw_path, upload_date) VALUES (?, ?, ?, ?)',
                (series_uid, mhd_path, raw_path, upload_date)
            )
        print(f"已记录: series_uid={series_uid}")
    except sqlite3.Error as e:
        print(f"插入失败: {e}")

def get_ct_paths(series_uid):   #查询类操作需要游标（cursor）
    """根据series_uid查询MHD和RAW文件路径"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.execute(
                'SELECT mhd_path, raw_path FROM ct_data WHERE series_uid = ?',
                (series_uid,)
            )
            result = cursor.fetchone()
        return result if result else (None, None)
    except sqlite3.Error as e:
        print(f"查询失败: {e}")
        return (None, None)

def update_ct_record(series_uid, new_mhd_path=None, new_raw_path=None):
    """
    更新CT数据记录的MHD和RAW文件路径。
    """
    try:
        with sqlite3.connect(DB_PATH) as conn:
            updates = []
            params = []

            if new_mhd_path is not None:    #只更新 mhd_path
                updates.append('mhd_path = ?')
                params.append(new_mhd_path)

            if new_raw_path is not None:    #只更新 raw_path
                updates.append('raw_path = ?')
                params.append(new_raw_path)

            if not updates:
                print(f"未提供要更新的路径，series_uid={series_uid} 的记录未修改。")
                return

            sql_query = f"UPDATE ct_data SET {', '.join(updates)} WHERE series_uid = ?"
            params.append(series_uid)
            conn.execute(sql_query, tuple(params))

            if conn.total_changes > 0:
                print(f"已更新: series_uid={series_uid}")
            else:
                print(f"未找到或未更新记录: series_uid={series_uid}")
    except sqlite3.Error as e:
        print(f"更新失败: {e}")

def delete_ct_record(series_uid):
    """
    删除CT数据记录。
    """
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute('DELETE FROM ct_data WHERE series_uid = ?',
                         (series_uid,))

            if conn.total_changes > 0:  #判断是否有改动
                print(f"已删除: series_uid={series_uid}")
            else:
                print(f"未找到记录，无法删除: series_uid={series_uid}")
    except sqlite3.Error as e:
        print(f"删除失败: {e}")

def get_all_ct_records():
    """
    获取ct_data表中的所有记录。
    """
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.execute('SELECT series_uid, mhd_path, raw_path, upload_date FROM ct_data')
            all_records = cursor.fetchall()
        return all_records
    except sqlite3.Error as e:
        print(f"获取所有记录失败: {e}")
        return []

if __name__ == '__main__':
    setup_database()