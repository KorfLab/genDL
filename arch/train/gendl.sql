CREATE TABLE models (
	model_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
	depth INTEGER,
	sizes TEXT,
	lr REAL,
	batch INTEGER,
	epoch INTEGER,
	reg TEXT,
	dropout TEXT,
	model_h5 BLOB,
	data_id INTEGER
);

CREATE TABLE metrics (
	model_id INTEGER,
	data_id INTEGER,
	tpr REAL,
	tnr REAL,
	ppv REAL,
	npv REAL,
	fsc REAL,
	FOREIGN KEY ([model_id]) REFERENCES "models" ([model_id]) ON DELETE ACTION ON UPDATE NO ACTION
);

CREATE TABLE data (
	data_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
	type TEXT,
	level TEXT,
	number INTEGER,
	true_name TEXT,
	fake_name TEXT,
);

CREATE TABLE seq_features (
	id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
);