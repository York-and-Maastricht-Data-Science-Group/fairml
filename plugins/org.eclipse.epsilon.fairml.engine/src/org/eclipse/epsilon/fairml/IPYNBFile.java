/*********************************************************************
 * Copyright (c) 2021 The University of York.
 *
 * This program and the accompanying materials are made
 * available under the terms of the Eclipse Public License 2.0
 * which is available at https://www.eclipse.org/legal/epl-2.0/
 *
 * SPDX-License-Identifier: EPL-2.0
 *********************************************************************/
package org.eclipse.epsilon.fairml;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.List;

import org.apache.commons.lang3.CharUtils;
import org.apache.commons.lang3.StringUtils;

/**
 * @author Alfa Yohannis
 */
public class IPYNBFile {

	public static final String CSV_DELIMITER = ",";
	private static final char CSV_QUOTE = '"';
	private static final String CSV_QUOTE_STR = String.valueOf(CSV_QUOTE);
	private static final String CSV_ESCAPED_QUOTE_STR = CSV_QUOTE_STR + CSV_QUOTE_STR;
	private static final char[] CSV_SEARCH_CHARS =
			new char[] { CSV_DELIMITER.charAt(0), CSV_QUOTE, CharUtils.CR, CharUtils.LF };

	public static String escapeCSV(String cellValue) {
		if (StringUtils.containsNone(cellValue, CSV_SEARCH_CHARS)) {
			return cellValue;
		}
		return new StringBuilder()
				.append(CSV_QUOTE)
				.append(StringUtils.replace(cellValue, CSV_QUOTE_STR, CSV_ESCAPED_QUOTE_STR))
				.append(CSV_QUOTE)
				.toString();
	}

	protected String path;
	protected FairML fairML;
	protected PrintWriter pw;

	public IPYNBFile(String path, FairML fairML) {
		this.path = path;
		this.fairML = fairML;
	}

	public void save() throws FileNotFoundException {
		System.out.println("Exporting to IPYNB file ... finished!");
//		File file = new File(path);
//		file.getParentFile().mkdirs();
//		pw = new PrintWriter(file);
//		headerRecord(fairML.getColumnNames());
//		for (List<ValueWrapper> wrappers : fairML.getRows()) {
//			rowRecord(wrappers);
//		}
//		pw.close();
	}

	protected void addRecord(List<String> cellValues) {
		pw.println(String.join(CSV_DELIMITER, cellValues));
	}
}
