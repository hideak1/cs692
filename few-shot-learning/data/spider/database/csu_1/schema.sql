CREATE TABLE "Campuses" (
	"Id" INTEGER PRIMARY KEY, 
	"Campus" TEXT, 
	"Location" TEXT, 
	"County" TEXT, 
	"Year" INTEGER 
);

CREATE TABLE "csu_fees" ( 
	"Campus" INTEGER PRIMARY KEY, 
	"Year" INTEGER, 
	"CampusFee" INTEGER,
	FOREIGN KEY (Campus) REFERENCES Campuses(Id)
);

CREATE TABLE "degrees" ( 
	"Year" INTEGER,
	"Campus" INTEGER, 
	"Degrees" INTEGER,
	PRIMARY KEY (Year, Campus),
	FOREIGN KEY (Campus) REFERENCES Campuses(Id)
);



CREATE TABLE "discipline_enrollments" ( 
	"Campus" INTEGER, 
	"Discipline" INTEGER, 
	"Year" INTEGER, 
	"Undergraduate" INTEGER, 
	"Graduate" INTEGER,
	PRIMARY KEY (Campus, Discipline),
	FOREIGN KEY (Campus) REFERENCES Campuses(Id)
);



CREATE TABLE "enrollments" ( 
	"Campus" INTEGER, 
	"Year" INTEGER, 
	"TotalEnrollment_AY" INTEGER, 
	"FTE_AY" INTEGER,
	PRIMARY KEY(Campus, Year),
	FOREIGN KEY (Campus) REFERENCES Campuses(Id)
);

CREATE TABLE "faculty" ( 
	"Campus" INTEGER, 
	"Year" INTEGER, 
	"Faculty" REAL,
	FOREIGN KEY (Campus) REFERENCES Campuses(Id) 
);


