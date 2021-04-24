import './Dataset.css';
import MaterialTable from "material-table";
import {useEffect, useState} from "react";
import {getData} from "../../api/data";
import ModalImage from "react-modal-image";

const coverColumn = {
    title: 'cover',
    field: 'cover_url',
    render: item => {
        const url = "https://" + item.cover_url
        return <ModalImage
            small={url}
            large={url}
            alt={url}
        />;
    }
}

const datasetColumns = [
    'ym_track_id',
    'ym_album_id',
    'track_title',
    'artist_name',
    'audio_path',
    'lyrics_path',
    'cover_path',
    'cover_url',
    'emotion',
    'genre',
    'artist_country',
    'artist_year',
    'artist_age'
];

function Dataset({id}) {
    const [data, setData] = useState([]);

    useEffect(() => {
        getData().then(setData)
    }, [])

    return (
        <div id={id}>
            <MaterialTable
                columns={[coverColumn, ...datasetColumns.map(column => ({title: column, field: column}))]}
                data={data}
                title="Music tracks dataset"
                options={{
                    paging:true,
                    pageSize: 20,
                    emptyRowsWhenPaging: true,
                }}
            />
        </div>
    );
}

export default Dataset;
