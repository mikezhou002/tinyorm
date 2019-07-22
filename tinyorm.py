"""
    This is a tiny ORM
    ~~~~~~~~~~~~~~~~~~
    created time : 2019 07 21

    usage:

        >>> import sqlite3
        >>> db = sqlite3.connect(':memory:')
        >>> DB.init(db)
        >>> class Student(Model):
        ...     id = Integer(primary_key=True)
        ...     name = String(maxlength=50)
        ...     age = Integer()
        ...     def __repr__(self):
        ...         return '<Student:%s>' %(self.name)
        >>>
        >>> DB.execute(Student.__schema__) # doctest: +ELLIPSIS
        -1
        >>> bob = Student(name='Bob', age=23)
        >>> bob.save()
        1
        >>> bob = Student.get(1)
        >>> bob
        <Student:Bob>
        >>> lucy = Student(name='Lucy', age=20)
        >>> lucy.save()
        1
        >>> Student.find()
        [<Student:Bob>, <Student:Lucy>]
        >>> Student.find('where age < ?', 21)
        [<Student:Lucy>]
        >>> DB.close()

:copyright: (c) 2019 by Mike Zhou <yulyz2002@gmail.com>.
"""

from contextlib import contextmanager
from collections import Iterable
import logging
import re
import random

logger = logging.getLogger(__name__)
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def make_list(args):
    """ 辅助方法，确保返回一个可迭代对象 """
    if isinstance(args, str) or not isinstance(args, Iterable):
        return [args]
    return args

class DB:

    """
        DB是一个负责管理数据库连接，执行数据库查询的类。
        几乎都是类属性和类方法。

        主要类属性：
        db: 一个数据库连接对象。当db不为None,则不会启用连接池。
        _pool: 数据库连接池。
        poolsize: 连接池中连接数量上限。

    """

    db = None
    _pool = {
        'using':[],
        'unuse':[],
        'size':0
    }
    poolsize = 0

    @classmethod
    def init(cls, uri_or_db, poolsize=3):
        """
            初始化类
            :param uri_or_db 数据库连接URI，或者是一个已经打开的数据库连接
                             URI 格式，例如：mysql+cymysql://root:password@localhost:3306/database
                             如果传入的是已经打开的数据库连接，则会赋值给类的db属性，不会启用连接池。
            :param poolsize  连接池大小
        """
        if cls.db or cls._pool['size'] > 0:
            return

        # TODO: error handle
        if not isinstance(uri_or_db, str):
            # Not URI, so directly sighn the db connection to cls.db
            # Cant use pool in this situation
            cls.db = uri_or_db
            cls.dbtype = 'sqlite' if ('sqlite' in str(uri_or_db)) else 'other'
        elif uri_or_db.startswith('sqlite'):
            pattern = r'sqlite:///(.+)'
            cls.dbtype = 'sqlite'
            cls.db_path = re.search(pattern, uri_or_db).group(1)
            if cls.db_path == 'memory':
                cls.db_path = ':memory:'
        else:
            pattern = r'(?P<dbtype>.+?)\+(?P<backend>.+?)://(?P<user>.+?):(?P<password>.+?)@(?P<host>.+?):(?P<port>.+?)/(?P<database>.+)'
            r = re.search(pattern, uri_or_db)
            cls.dbtype = r.group('dbtype')
            cls.backend = r.group('backend')
            cls.user = r.group('user')
            cls.password = r.group('password')
            cls.host = r.group('host')
            cls.port = int(r.group('port'))
            cls.database = r.group('database')
        cls.plh = '?' if cls.dbtype == 'sqlite' else '%s'
        if isinstance(poolsize, int) and poolsize >=0:
            cls.poolsize = int(poolsize)
        else:
            raise TypeError('poolsize must be int and not smaller than 0')


    @classmethod
    def select(cls, sql, args=None, size=None):
        """
            使用类管理的数据库连接对象查询数据库
            :param sql sql语句，语句中的参数用"?"占位
            :args 元组，对应语句中的参数用"?"占位
            :returns 字典型（键为表字段名）的记录集
        """
        args = () if args is None else make_list(args)
        sql = sql.replace('?', cls.plh)
        logger.info(f'<select>{sql}, {args}')
        with cls.oncursor() as cursor:
            cursor.execute(sql, args)
            if size:
                rs = cursor.fetchmany(size)
            else:
                rs = cursor.fetchall()
            rs = cls._dictfy(cursor, rs)
            return rs

    @classmethod
    def execute(cls, sql, args=None):
        sql = sql.replace('?', cls.plh)
        args = () if args is None else make_list(args)
        logger.info(f'<execute>{sql}, {args}')
        with cls.oncursor() as cursor:
            cursor.execute(sql, args)
            return cursor.rowcount

    @classmethod
    def oncursor(cls):
        """
            returns: 返回上下文管理器。
                    该上下文管理器从资源池中获取一个可用的连接，生成cursor
                    并自动commit（若出错则rollback）
                    自动返还连接
        """
        return CursorContext(cls)

    @classmethod
    def close(cls):
        """关闭所引用的所有数据库连接"""
        try:
            if cls.db:
                cls.db.close()
            for db in cls._pool['using']:
                db.close()
            for db in cls._pool['unuse']:
                db.close()
        except:
            pass

    @classmethod
    def get_db(cls):
        """ 获取一个数据库连接，若启用资源池，则从资源池中拿取 """
        if cls.db is not None:
            return cls.db
        cls._connect()

        # No need use pool
        if cls.poolsize <= 1:
            cls.db = cls._pool['unuse'][0]
            return cls.db

        if cls._pool['unuse']:
            db = random.choice(cls._pool['unuse'])
            cls._pool['unuse'].remove(db)
            cls._pool['using'].append(db)
            return db
        else:
            return random.choice(cls._pool['unuse'])

    @classmethod
    def put_db(cls, db):
        """ 返还连接 """
        if cls.db:
            return
        cls._pool['using'].remove(db)
        cls._pool['unuse'].append(db)


    @classmethod
    def _connect(cls):
        """ 建立连接。若启用资源池，则自动将连接置入资源池。 """
        if cls.poolsize !=0 and cls._pool['size'] == cls.poolsize:
            return

        if cls.dbtype == 'sqlite':
            import sqlite3
            db = sqlite3.connect(cls.db_path)

        else:
            try:
                backend = __import__(cls.backend)
                db = backend.connect(host=cls.host,
                                        port=int(cls.port),
                                        user=cls.user,
                                        passwd=cls.password,
                                        db=cls.database)
            except Exception as e:
                db= backend.connect(host=cls.host,
                                        port=int(cls.port),
                                        user=cls.user,
                                        password=cls.password,
                                        database=cls.database)
            except Exception as e:
                db = backend.connect(host=cls.host,
                                        port=int(cls.port),
                                        user=cls.user,
                                        passwd=cls.password,
                                        database=cls.database)
            except Exception as e:
                raise RuntimeError('Can not connect database')

        logger.info(f"creating db connect<{id(db)}>, poolsize:{cls._pool['size']}")
        cls._pool['unuse'].append(db)
        cls._pool['size'] += 1

    @classmethod
    def _dictfy(cls, cursor, rs):
        """工具方法， 将从数据库返回的元组列表转换成字典"""
        keys = [item[0] for item in cursor.description]
        return [dict(zip(keys, r)) for r in rs]

    @classmethod
    def create_table(cls, model):
        """ 根据模型在数据库中建表 """
        if cls.dbtype == 'mysql':
            schema = model.__schema__.replace('AUTOINCREMENT', 'AUTO_INCREMENT')
            schema = schema.replace(';' ,'ENGINE=InnoDB DEFAULT CHARSET=utf8;')
        cls.execute(schema)

    @classmethod
    def drop_table(cls, model):
        """ 删除指定模型在数据库中的表 """
        sql = F'DROP TABLE {model.__table__}'
        cls.execute(sql)



class CursorContext:
    """ cursor上下文管理器
        实现获取连接-->commit-->返还连接自动化
    """
    def __init__(self, DB):
        self.DB = DB

    def __enter__(self):
        self.db = self.DB.get_db()
        self.cursor = self.db.cursor()
        logger.info(f'DB:{id(self.DB)}-db:{id(self.db)}-cursor:{id(self.cursor)}')
        return self.cursor

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.db.rollback()
        self.db.commit()
        self.DB.put_db(self.db)
        self.cursor.close()


def placeholders(fields, kv=False):
    """ 工具方法，通过字段列表生成相应的 '? , ?, ?' 占位符 或者 'field=?' 占位符 """
    if kv:
        return ', '.join([f'{field} = ?' for field in fields])
    return ', '.join(['?' for _ in fields])

class Field(object):
    def __init__(self, ddl=None, primary_key=False, nullable=False, default=None):
        self.ddl = ddl
        self.primary_key = primary_key
        self.nullable = nullable
        self.default = default
    def __repr__(self):
        echo = f'<{self.ddl}'
        echo += ' primary_key' if self.primary_key else ''
        echo += '>'
        return echo

class String(Field):
    def __init__(self, primary_key=False, nullable=True, default=None, maxlength=None):
        if maxlength is None:
            raise ValueError('maxlength can not be null')
        self.maxlength = maxlength
        ddl = 'VARCHAR(' + str(maxlength) + ')'
        super().__init__(ddl, primary_key, nullable, default)


class Integer(Field):
    def __init__(self, primary_key=False, nullable=True, default=None):
        ddl = 'INTEGER'
        super().__init__(ddl, primary_key, nullable, default)


class Float(Field):
    def __init__(self, primary_key=False, nullable=True, default=None):
        ddl = 'FLOAT'
        super().__init__(ddl, primary_key, nullable, default)


class Text(Field):
    def __init__(self, primary_key=False, nullable=True, default=None):
        ddl = 'TEXT'
        super().__init__(ddl, primary_key, nullable, default)

class Bool(Field):
    def __init__(self, primary_key=False, nullable=True, default=None):
        ddl = 'BOOLEAN'
        super().__init__(ddl, primary_key, nullable, default)


def create_schema(table_name, mapping):
    """ 生成建表SQL
        :param table_name: 表名
        :param mapping: {字段名:字段} 的字典
    """
    schema = f'CREATE TABLE IF NOT EXISTS {table_name} ('
    ddls = []
    for k, v in mapping.items():
        field_ddl = f'{k} {v.ddl}'
        if v.primary_key:
            field_ddl += ' PRIMARY KEY'
            if isinstance(v, Integer):
                field_ddl += ' AUTOINCREMENT'
            ddls.append(field_ddl)
            continue
        if not v.nullable:
            field_ddl += ' NOT NULL'
        if v.default:
            field_ddl += f' DEFAULT {v.default}'
        ddls.append(field_ddl)
    schema += ','.join(ddls)
    schema +=  ');'
    return schema


class ModelMeta(type):

    """ metaclass, 获得子类中与SQL有关的信息。并将Field类属性删除，避免对子类的实例属性造成影响。 """
    def __new__(cls, name, bases, attrs):
        if  'Model' in name:
            return super().__new__(cls, name, bases, attrs)

        table_name = attrs.get('__table__') or name.lower()
        logger.info(f'Found table:{table_name}')

        fields = []
        primary_key = None
        mapping = {}

        for k, v in attrs.items():
            if isinstance(v, Field):
                mapping[k] = v
                if v.primary_key:
                    if primary_key is None:
                        primary_key = k
                        logger.info(f'Fund primary key:{k}:{v.ddl}')
                    else:
                        raise ValueError('Duplicated primary key')
                    continue
                fields.append(k)
                logger.info(f'Fund Field:{k}:{v.ddl}')
        if primary_key is None:
            raise ValueError('Primary key required')
        for k in mapping:
            attrs.pop(k)

        attrs['__table__'] = table_name
        attrs['__primary_key__'] = primary_key
        attrs['__fields__'] = fields
        attrs['__mapping__'] = mapping
        attrs['__select__'] = f"SELECT {', '.join(fields)}, {primary_key} FROM {table_name} "

        # For compatible mysql, need to deel with None value in sql
        attrs['__insert__'] = f"INSERT INTO {table_name} "
        attrs['__update__'] = f"UPDATE {table_name} SET {placeholders(fields, kv=True)} WHERE {placeholders(primary_key, kv=True)}"
        attrs['__delete__'] = f"DELETE FROM {table_name} WHERE {placeholders(primary_key, kv=True)}"
        attrs['__schema__'] = create_schema(table_name, mapping)

        return super().__new__(cls, name, bases, attrs)


class Model(object, metaclass=ModelMeta):

    """ 模型类，通过元类获得的数据创建需要数据库支撑的方法，可供子类直接调用 """

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def _getattr(self, attr):
        return getattr(self, attr, None)

    @classmethod
    def get(cls, pk):
        """通过主键从数据库中查寻数据，返回结果封装成类实例"""
        sql = cls.__select__ + f"WHERE {cls.__primary_key__} = ?"
        rs = DB.select(sql, [pk], 1)
        if rs:
            return cls(**rs[0])

    @classmethod
    def find(cls, filters=None, args=None, size=None):
        """
        查询。
        Example: Student.find('where age >? limit ?', (21, 4))
        """
        sql = cls.__select__
        if filters:
            sql +=  filters
        rs = DB.select(sql, args, size)
        if rs:
            return [cls(**r) for r in rs]

    @property
    def clean_fields(self):
        """
            获取值不为None的fields
        """
        fields = [field for field in self.__fields__ if self._getattr(field) is not None]
        if self._getattr(self.__primary_key__) is not None:
            fields.append(self.__primary_key__)
        return fields

    @property
    def clean_values(self):
        """
            不为None的所有字段值
        """
        args = [self._getattr(field) for field in self.clean_fields]
        if self._getattr(self.__primary_key__) is not None:
            args.append(self._getattr(self.__primary_key__))
        return args

    def save(self):
        """
            将类实例写入数据库，即在数据库插入1行
        """
        args = self.clean_values
        sql = self.__insert__ + f"({','.join(self.clean_fields)}) VALUES ({placeholders(args)})"
        return DB.execute(sql, args)

    def update(self):
        """
            在数据库中更新类实例对应的行
        """
        sql = self.__update__
        args = [self._getattr(field) for field in self.__fields__] + [self._getattr(__primary_key__)]
        return DB.execute(sql, args)

    def delete(self):
        """
            在数据库中删除类实例对应的行
        """
        sql = self.__delete__
        args = [self._getattr(self.__primary_key__)]
        return DB.execute(sql, args)
