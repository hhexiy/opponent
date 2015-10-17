require 'util.QAMinibatchLoader'

local QBMinibatchLoader = {}
QBMinibatchLoader.__index = QBMinibatchLoader
setmetatable(QBMinibatchLoader, {__index = QAMinibatchLoader})
