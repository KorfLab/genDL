CREATE TABLE models (
	model_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
	depth INTEGER,
	sizes TEXT,
	lr REAL,
	batch INTEGER,
	epoch INTEGER,
	reg TEXT,
	dropout TEXT,
);

CREATE TABLE metrics (
	model_id INTEGER,
	tpr REAL,
	tnr REAL,
	ppv REAL,
	npv REAL,
	fsc REAL,
	FOREIGN KEY ([model_id]) REFERENCES "models" ([model_id]) ON DELETE ACTION ON UPDATE NO ACTION
);

CREATE TABLE data (
	model_id INTEGER,
	type TEXT,
	level TEXT,
	number INTEGER,
	true_name TEXT,
	fake_name TEXT,
	FOREIGN KEY ([model_id]) REFERENCES "models" ([model_id]) ON DELETE ACTION ON UPDATE NO ACTION
);
