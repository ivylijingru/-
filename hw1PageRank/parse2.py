import xml.sax as sax
import xml.sax.handler as saxhandler
import pprint

class TagParser(saxhandler.ContentHandler):
    # http://docs.python.org/library/xml.sax.handler.html#contenthandler-objects
    def __init__(self):
        self.tags = {}
    def startElement(self, name, attrs):
        self.tag = name
    def endElement(self, name):
        if self.tag:
            self.tags[self.tag] = self.data
            self.tag = None
            self.data = None
    def characters(self, content):
        self.data = content

parser = TagParser()
src = '''\
<some_root_name>
    <tag_x>bubbles</tag_x>
    <tag_y>car</tag_y>
    <tag...>42</tag...>
</some_root_name>'''
sax.parseString(src, parser)
pprint.pprint(parser.tags)