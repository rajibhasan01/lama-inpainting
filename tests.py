"""
This file demonstrates common uses for the Python unittest module with Flask

Documentation:

* https://docs.python.org/3/library/unittest.html
* http://flask.pocoo.org/docs/latest/testing/
"""
import random
import unittest
import main


class FlaskTestCase(unittest.TestCase):
    """ This is one of potentially many TestCases """

    def setUp(self):
        app = main.create_app()
        app.debug = True
        self.app = app.test_client()

    def test_route_hello_world(self):
        res = self.app.get("/")
        # print(dir(res), res.status_code)
        assert res.status_code == 200
        assert b"Hello World" in res.data

    def test_route_foo(self):
        res = self.app.get("/foo/12345")
        assert res.status_code == 200
        assert b"123456" in res.data


if __name__ == '__main__':
    unittest.main()
